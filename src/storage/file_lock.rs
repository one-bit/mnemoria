use fs2::FileExt;
use std::fs::{File, OpenOptions};
use std::path::{Path, PathBuf};

const LOCK_FILENAME: &str = "mnemoria.lock";

/// A file-based advisory lock using `flock(2)`.
///
/// - **Exclusive** (`LOCK_EX`): held during writes. Blocks other writers and readers.
/// - **Shared** (`LOCK_SH`): held during reads. Multiple readers can hold it concurrently;
///   blocks only while an exclusive lock is held.
///
/// The lock is released when the guard is dropped (closing the file descriptor).
/// If the process crashes, the OS releases the lock automatically.
pub struct FileLock {
    lock_path: PathBuf,
}

/// RAII guard that releases the flock when dropped.
pub struct FileLockGuard {
    _file: File,
}

impl Drop for FileLockGuard {
    fn drop(&mut self) {
        // Closing the File releases the flock. The explicit unlock
        // call is belt-and-suspenders in case the Drop order is unusual.
        let _ = self._file.unlock();
    }
}

impl FileLock {
    pub fn new(base_path: &Path) -> Result<Self, crate::Error> {
        let lock_path = base_path.join(LOCK_FILENAME);

        // Create the lock file if it doesn't exist. The file is never
        // deleted; its contents are irrelevant.
        let _file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(false)
            .open(&lock_path)?;

        Ok(Self { lock_path })
    }

    /// Acquire an exclusive (writer) lock. Blocks until available.
    pub fn lock_exclusive(&self) -> Result<FileLockGuard, crate::Error> {
        let file = File::open(&self.lock_path)?;
        file.lock_exclusive()?;
        Ok(FileLockGuard { _file: file })
    }

    /// Acquire a shared (reader) lock. Blocks only while an exclusive lock is held.
    pub fn lock_shared(&self) -> Result<FileLockGuard, crate::Error> {
        let file = File::open(&self.lock_path)?;
        file.lock_shared()?;
        Ok(FileLockGuard { _file: file })
    }
}
