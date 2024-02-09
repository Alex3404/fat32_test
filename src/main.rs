#![feature(ascii_char, ascii_char_variants)]
mod disk;
mod fat;
mod filesystem;

// TODO Break everything down into structs
// (Disk_W, Disk_R into their own struct)
// (FSInfo, Fats, FatVolumeHeader into their own struct)
// TODO Add a interface between disk and fat32 cluster system
// TODO Add functions for writing specific bytes offsets into sectors at specific byte lengths
// TODO do MORE ABSTRACTION! YAY

use std::fmt::Write;

use disk::Disk;
use fat::Fat32FileSystem;
use filesystem::{FileAttributes, FileSystem, WriteMode};

const DISK_IMAGE_FILE: &str = "disk.img";

fn main() {
    let disk = Disk::new(DISK_IMAGE_FILE);
    let mut fat32 = Fat32FileSystem::create_from_disk(disk, 0_usize);
    let file = fat32.load_file("HELLO.TXT", "FOLDER");
    match file {
        Ok(file) => println!("Found file"),
        Err(error) => panic!("Error: {}", error.0),
    }

    let file = fat32.create_file("NEW.TXT", "", FileAttributes::NONE);
    let file = match file {
        Ok(file) => {
            println!("Created file");
            file
        }
        Err(error) => panic!("Error: {}", error.0),
    };

    let write_result = fat32.write_to_file(file, "Hello World!".as_bytes(), WriteMode::OVERWRITE);
    match write_result {
        Ok(_) => println!("Wrote to file!"),
        Err(error) => panic!("Write Error: {}", error.0),
    };
}
