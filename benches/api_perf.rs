use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use mnemoria::storage::{Manifest, log_reader};
use mnemoria::{Config, DurabilityMode, EntryType, Mnemoria, TimelineOptions};
use std::hint::black_box;
use tempfile::TempDir;
use tokio::runtime::Runtime;

const ENTRY_COUNTS: &[usize] = &[1_000, 5_000];

fn seed_memory(rt: &Runtime, count: usize) -> TempDir {
    seed_memory_with_durability(rt, count, DurabilityMode::None)
}

fn seed_memory_with_durability(rt: &Runtime, count: usize, durability: DurabilityMode) -> TempDir {
    let temp_dir = tempfile::tempdir().expect("failed to create tempdir");
    let config = Config {
        durability,
        ..Config::default()
    };
    let memory = rt
        .block_on(Mnemoria::create_with_config(temp_dir.path(), config))
        .expect("failed to create memory");

    for i in 0..count {
        let summary = format!("summary {i}");
        let content = if i % 10 == 0 {
            format!("entry {i} contains rust perf query token")
        } else {
            format!("entry {i} filler content")
        };

        rt.block_on(memory.remember(EntryType::Discovery, &summary, &content))
            .expect("failed to add entry");
    }

    temp_dir
}

fn bench_search_memory(c: &mut Criterion) {
    let rt = Runtime::new().expect("failed to build runtime");
    let mut group = c.benchmark_group("search_memory");

    for &count in ENTRY_COUNTS {
        let temp_dir = seed_memory(&rt, count);
        let memory = rt
            .block_on(Mnemoria::open(temp_dir.path()))
            .expect("failed to open memory");

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, _| {
            b.iter(|| {
                rt.block_on(memory.search_memory("rust perf query", 10))
                    .expect("search failed");
            });
        });

        drop(memory);
        drop(temp_dir);
    }

    group.finish();
}

fn bench_timeline_cached(c: &mut Criterion) {
    let rt = Runtime::new().expect("failed to build runtime");
    let mut group = c.benchmark_group("timeline_cached");

    for &count in ENTRY_COUNTS {
        let temp_dir = seed_memory(&rt, count);
        let memory = rt
            .block_on(Mnemoria::open(temp_dir.path()))
            .expect("failed to open memory");

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, _| {
            b.iter(|| {
                rt.block_on(memory.timeline(TimelineOptions {
                    limit: 100,
                    since: None,
                    until: None,
                    reverse: true,
                }))
                .expect("timeline failed");
            });
        });

        drop(memory);
        drop(temp_dir);
    }

    group.finish();
}

fn bench_timeline_disk_scan(c: &mut Criterion) {
    let rt = Runtime::new().expect("failed to build runtime");
    let mut group = c.benchmark_group("timeline_disk_scan");

    for &count in ENTRY_COUNTS {
        let temp_dir = seed_memory(&rt, count);
        let log_path = Manifest::log_path(temp_dir.path());

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, _| {
            b.iter(|| {
                let mut entries = log_reader::read_all(&log_path).expect("timeline baseline read");
                entries.reverse();
                entries.truncate(100);
                black_box(entries);
            });
        });

        drop(temp_dir);
    }

    group.finish();
}

fn bench_get_cached(c: &mut Criterion) {
    let rt = Runtime::new().expect("failed to build runtime");
    let mut group = c.benchmark_group("get_cached");

    for &count in ENTRY_COUNTS {
        let temp_dir = tempfile::tempdir().expect("failed to create tempdir");
        let ids = {
            let memory = rt
                .block_on(Mnemoria::create(temp_dir.path()))
                .expect("failed to create memory");

            let mut ids = Vec::with_capacity(count);
            for i in 0..count {
                let summary = format!("summary {i}");
                let content = format!("content {i}");
                let id = rt
                    .block_on(memory.remember(EntryType::Discovery, &summary, &content))
                    .expect("failed to add entry");
                ids.push(id);
            }

            ids
        };

        let memory = rt
            .block_on(Mnemoria::open(temp_dir.path()))
            .expect("failed to open memory");
        let target_id = ids[count / 2].clone();

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, _| {
            b.iter(|| {
                rt.block_on(memory.get(&target_id)).expect("get failed");
            });
        });

        drop(memory);
        drop(temp_dir);
    }

    group.finish();
}

fn bench_get_disk_scan(c: &mut Criterion) {
    let rt = Runtime::new().expect("failed to build runtime");
    let mut group = c.benchmark_group("get_disk_scan");

    for &count in ENTRY_COUNTS {
        let temp_dir = tempfile::tempdir().expect("failed to create tempdir");
        let ids = {
            let memory = rt
                .block_on(Mnemoria::create(temp_dir.path()))
                .expect("failed to create memory");

            let mut ids = Vec::with_capacity(count);
            for i in 0..count {
                let summary = format!("summary {i}");
                let content = format!("content {i}");
                let id = rt
                    .block_on(memory.remember(EntryType::Discovery, &summary, &content))
                    .expect("failed to add entry");
                ids.push(id);
            }

            ids
        };

        let target_id = ids[count / 2].clone();
        let log_path = Manifest::log_path(temp_dir.path());

        group.throughput(Throughput::Elements(count as u64));
        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, _| {
            b.iter(|| {
                let entries = log_reader::read_all(&log_path).expect("get baseline read");
                let entry = entries.into_iter().find(|e| e.id == target_id);
                black_box(entry);
            });
        });

        drop(temp_dir);
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Write-throughput benchmarks: entries/sec across durability modes
// ---------------------------------------------------------------------------

const WRITE_BATCH: usize = 200;

fn bench_write_throughput(c: &mut Criterion) {
    let rt = Runtime::new().expect("failed to build runtime");
    let mut group = c.benchmark_group("write_throughput");
    group.sample_size(10);

    let modes: &[(&str, DurabilityMode)] = &[
        ("fsync", DurabilityMode::Fsync),
        ("flush_only", DurabilityMode::FlushOnly),
        ("none", DurabilityMode::None),
    ];

    for &(label, mode) in modes {
        group.throughput(Throughput::Elements(WRITE_BATCH as u64));
        group.bench_with_input(BenchmarkId::from_parameter(label), &mode, |b, &mode| {
            b.iter_with_setup(
                || {
                    let temp_dir = tempfile::tempdir().expect("failed to create tempdir");
                    let config = Config {
                        durability: mode,
                        ..Config::default()
                    };
                    let memory = rt
                        .block_on(Mnemoria::create_with_config(temp_dir.path(), config))
                        .expect("failed to create memory");
                    (temp_dir, memory)
                },
                |(_temp_dir, memory)| {
                    for i in 0..WRITE_BATCH {
                        rt.block_on(memory.remember(
                            EntryType::Discovery,
                            &format!("bench entry {i}"),
                            &format!("bench content for entry {i}"),
                        ))
                        .expect("remember failed");
                    }
                },
            );
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Query latency benchmarks: BM25-only search at various store sizes
// ---------------------------------------------------------------------------

fn bench_search_latency(c: &mut Criterion) {
    let rt = Runtime::new().expect("failed to build runtime");
    let mut group = c.benchmark_group("search_latency");
    group.sample_size(30);

    let sizes: &[usize] = &[1_000, 5_000, 10_000];

    for &count in sizes {
        let temp_dir = seed_memory(&rt, count);
        let memory = rt
            .block_on(Mnemoria::open(temp_dir.path()))
            .expect("failed to open memory");

        group.bench_with_input(BenchmarkId::from_parameter(count), &count, |b, _| {
            b.iter(|| {
                rt.block_on(memory.search_memory("rust perf query", 10))
                    .expect("search failed");
            });
        });

        drop(memory);
        drop(temp_dir);
    }

    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default().sample_size(30)
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = bench_search_memory, bench_timeline_cached, bench_timeline_disk_scan, bench_get_cached, bench_get_disk_scan, bench_write_throughput, bench_search_latency
}
criterion_main!(benches);
