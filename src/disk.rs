use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};

pub struct Disk {
    disk: File,
}

impl Disk {
    pub fn new(file_path: &str) -> Self {
        Self {
            disk: OpenOptions::new()
                .write(true)
                .read(true)
                .open(file_path)
                .unwrap(),
        }
    }

    pub fn write_sectors_to_disk(
        &mut self,
        lba: usize,
        _count: usize,
        sector_size: usize,
        buffer: &[u8],
    ) {
        println!(
            "Writing LBA: {}, Sector Count: {}, Bytes {}",
            lba,
            _count,
            buffer.len()
        );
        let _ = self
            .disk
            .seek(SeekFrom::Start(lba as u64 * sector_size as u64))
            .unwrap();
        self.disk.write_all(buffer).unwrap();
    }

    pub fn read_sectors_from_disk(
        &mut self,
        lba: usize,
        _count: usize,
        sector_size: usize,
        buffer: &mut [u8],
    ) {
        let _ = self
            .disk
            .seek(SeekFrom::Start(lba as u64 * sector_size as u64))
            .unwrap();
        println!(
            "Reading LBA: {}, Sector Count: {}, Bytes {}",
            lba,
            _count,
            buffer.len()
        );
        let _ = self.disk.read_exact(buffer);
    }
}
