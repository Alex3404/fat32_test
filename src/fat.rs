use crate::disk::Disk;
use crate::filesystem::{self, FileSystemError};
use core::mem;
use std::ascii;
use std::borrow::Borrow;
use std::marker::PhantomData;
use std::ops::Add;
use std::slice::ChunksExact;
extern crate static_assertions as sa;

fn str_to_padded_ascii_string<const LENGTH: usize>(
    string: &str,
    pad_character: ascii::Char,
) -> [u8; LENGTH] {
    let ascii = string.as_ascii();
    let mut ascii = ascii.map(Vec::from).unwrap_or_default();

    ascii.resize(LENGTH, pad_character);

    let ascii: &[u8] = ascii.as_bytes();
    ascii.try_into().unwrap_or([0_u8; LENGTH])
}

fn append_ascii_to_string<const LENGTH: usize>(
    string: &mut String,
    ascii: &[u8; LENGTH],
    pad_char: ascii::Char,
) {
    let pad_byte = pad_char.to_u8();

    // Find the last index in a string where all the chars infront is the pad char
    let last_padded_inverse_index = ascii
        .iter()
        .rev()
        .enumerate()
        .scan(None, |state, (index, &current)| {
            let prev_char: u8 = match *state {
                Some(val) => val,
                None => {
                    *state = Some(current);
                    return Some(index);
                }
            };

            if prev_char == pad_byte && current != pad_byte {
                None
            } else {
                Some(index)
            }
        })
        .last();

    let last_pad = match last_padded_inverse_index {
        Some(last_padded_inverse_index) => ascii.len() - (last_padded_inverse_index + 1),
        None => LENGTH,
    };

    let ascii = &ascii[..last_pad];

    let str = String::from_utf8(Vec::from(ascii));
    if let Ok(str) = str {
        string.push_str(&str)
    }
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct FSInfo {
    lead_sig: u32, // Should be 0x41615252
    reserved_1: [u8; 480],
    second_sig: u32,                     // Should be 0x61417272
    last_known_free_cluster_count: u32,  // if 0xFFFFFFFF must be computed
    available_clusters_start_index: u32, // if 0xFFFFFFFF start cluster when looking for avaliable clusters
    reserved_2: [u8; 12],
    trail_sig: u32, // Should be 0xAA550000
}

#[repr(packed)]
#[derive(Clone, Copy, PartialEq)]
pub struct DirectoryAttributes(pub u8);
impl DirectoryAttributes {
    const READ_ONLY: u8 = 0x01;
    const HIDDEN: u8 = 0x02;
    const SYSTEM: u8 = 0x04;
    const VOLUME_ID: u8 = 0x08;
    const DIRECTORY: u8 = 0x10;
    const ARCHIVE: u8 = 0x20;
    const FILE: u8 = 0x00;

    const LONG_FILE_NAME: u8 = Self::READ_ONLY | Self::HIDDEN | Self::SYSTEM | Self::VOLUME_ID;

    fn is_read_only(&self) -> bool {
        self.0 & Self::READ_ONLY == Self::READ_ONLY
    }

    fn is_hidden(&self) -> bool {
        self.0 & Self::HIDDEN == Self::HIDDEN
    }

    fn is_system(&self) -> bool {
        self.0 & Self::SYSTEM == Self::SYSTEM
    }

    fn is_volume_id(&self) -> bool {
        self.0 & Self::VOLUME_ID == Self::VOLUME_ID
    }

    fn is_directory(&self) -> bool {
        self.0 & Self::DIRECTORY == Self::DIRECTORY
    }

    fn is_file(&self) -> bool {
        self.0 & Self::DIRECTORY != Self::DIRECTORY
    }

    fn is_archive(&self) -> bool {
        self.0 & Self::ARCHIVE == Self::ARCHIVE
    }

    fn has_long_file_name(&self) -> bool {
        self.0 & Self::LONG_FILE_NAME == Self::LONG_FILE_NAME
    }
}
impl TryFrom<u8> for DirectoryAttributes {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        Ok(Self(value))
    }
}

#[repr(packed)]
#[derive(Clone, Copy)]
pub struct U16time(u16);
impl U16time {}
impl TryFrom<u16> for U16time {
    type Error = ();

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        Ok(Self(value))
    }
}

#[repr(packed)]
#[derive(Clone, Copy)]
pub struct U16date(u16);
impl U16date {}
impl TryFrom<u16> for U16date {
    type Error = ();

    fn try_from(value: u16) -> Result<Self, Self::Error> {
        Ok(Self(value))
    }
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
pub struct DirectoryEntry {
    file_name: [u8; 8],
    extension: [u8; 3],
    attributes: DirectoryAttributes,
    reserved: u8,

    creation_time_hundreth_of_second: u8,
    creation_time: U16time,
    creation_date: U16date,
    last_access_date: U16date,

    high_bits_of_cluster_number: u16,

    last_modification_time: U16time,
    last_modification_date: U16date,

    low_bits_of_cluster_number: u16,
    size_of_file: u32,
}

impl DirectoryEntry {
    fn get_cluster(&self) -> usize {
        (self.high_bits_of_cluster_number as usize) << 16
            | (self.low_bits_of_cluster_number as usize)
    }

    fn new(
        file_name: [u8; 8],
        extension: [u8; 3],
        cluster: u32,
        size_of_file: u32,
        attributes: DirectoryAttributes,
    ) -> Self {
        Self {
            file_name,
            extension,
            attributes,
            reserved: 0x0,
            creation_time_hundreth_of_second: 0x0,
            creation_time: U16time(0x0000),
            creation_date: U16date(0x0000),
            last_access_date: U16date(0x0000),
            high_bits_of_cluster_number: (size_of_file >> 16) as u16,
            last_modification_date: U16date(0x0000),
            last_modification_time: U16time(0x0000),
            low_bits_of_cluster_number: (size_of_file & 0xFFFF) as u16,
            size_of_file,
        }
    }
}

#[repr(C, packed)]
#[derive(Clone, Copy, PartialEq)]
pub struct LongFileNameEntry {
    idk: u8,
    first: [u16; 5],
    attribute: DirectoryAttributes, // Always equal to DirectoryAttributes::LONG_FILE_NAME
    entry_type: u8,
    checksum_of_short_filename: u8,
    next: [u16; 6],
    zero: u16,
    last: [u16; 2],
}

#[repr(C, packed)]
#[derive(Clone, Copy)]
struct FAT32VolumeHeader {
    jump_instructions: [u8; 3],
    oem_identifier: [u8; 8],

    bytes_per_sector: u16,
    sectors_per_cluster: u8,
    reserved_sectors: u16,
    fat_count: u8,
    root_directory_count: u16,
    total_sectors: u16,
    media_descriptor: u8,
    sectors_per_fat: u16, // Sectors per fat (For Fat16)

    sectors_per_track: u16,
    number_of_heads: u16,
    beginning_lba: u32, // Hidden Sectors
    large_total_sectors: u32,

    large_sectors_per_fat: u32,
    mirroring_flags: u16,
    version: u16,
    root_directory_cluster: u32,
    location_of_fs_info_sector: u16,
    location_of_backup_sector: u16,
    boot_file_name: [u8; 12],
    physical_drive_number: u8,
    extra_flags: u8,
    extended_boot_sig: u8,
    volume_serial_number: u32,
    volume_label: [u8; 11],
    system_identifier_str: [u8; 8],
    padding: [u8; 420],
    bootable_partition_sig: u16,
}

pub struct Fat32FileSystem {
    volume_header: FAT32VolumeHeader,
    fs_info: FSInfo,
    fat: Vec<u32>,
    disk: Disk,
}
trait Cluster<T: Into<usize> = Self>: Into<usize> + From<usize> {}

#[derive(Clone, Copy, PartialEq, Eq)]
struct DirectoryCluster(usize);

impl Cluster for DirectoryCluster {}

impl From<usize> for DirectoryCluster {
    fn from(value: usize) -> Self {
        // This assumes that the cluster being passed in is valid
        Self(value)
    }
}

impl Into<usize> for DirectoryCluster {
    fn into(self) -> usize {
        self.0
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct DataCluster(usize);

impl Cluster for DataCluster {}

impl From<usize> for DataCluster {
    fn from(value: usize) -> Self {
        // This assumes that the cluster being passed in is valid
        Self(value)
    }
}

impl Into<usize> for DataCluster {
    fn into(self) -> usize {
        self.0
    }
}

struct ClustersIter<'a, T> {
    start_cluster: usize,
    current_cluster: usize,
    file_system: &'a Fat32FileSystem,
    _phantom: PhantomData<T>,
}

impl<'a, T> ClustersIter<'a, T>
where
    T: Into<usize>,
{
    fn new(file_system: &'a Fat32FileSystem, cluster: T) -> Self {
        let cluster: usize = T::into(cluster);
        Self {
            start_cluster: cluster,
            current_cluster: cluster,
            file_system,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T> Iterator for ClustersIter<'a, T>
where
    T: From<usize>,
{
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        let next_cluster = self.file_system.fat[self.current_cluster] as usize;

        if next_cluster >= self.file_system.min_cluster()
            && next_cluster <= self.file_system.max_cluster()
        {
            self.current_cluster = next_cluster;
            Some(T::from(next_cluster))
        } else {
            None
        }
    }
}

pub struct Fat32FileEntry {
    directory_entry: Option<DirectoryEntry>,
    long_file_name_entry: Option<LongFileNameEntry>,
}

impl Fat32FileEntry {
    fn get_file_name(&self) -> Option<String> {
        let mut file_name = String::new();
        match self.directory_entry {
            Some(entry) => {
                append_ascii_to_string(&mut file_name, &entry.file_name, ascii::Char::Space);

                // TODO: Construct long file name string here

                Some(file_name)
            }
            None => return None,
        }
    }
}

pub struct Fat32Entires<'a, 'b> {
    file_system: &'a Fat32FileSystem,
    current_cluster: Option<DirectoryCluster>,
    cluster_data: Vec<u8>,
    chunked_iter: ChunksExact<'b, u8>,
    cluster_iter: ClustersIter<'a, DirectoryCluster>,
}

impl Fat32Entires<'_, '_> {
    fn new<'a, 'b>(file_system: &'a Fat32FileSystem, directory_cluster: DirectoryCluster) -> Self {
        let cluster_data = vec![0_u8; file_system.size_of_cluster()];

        Self {
            file_system,
            current_cluster: None,
            cluster_data,
            chunked_iter: cluster_data.chunks_exact(Fat32FileSystem::size_of_directory_entry()),
            cluster_iter: ClustersIter::new(file_system, directory_cluster),
        }
    }

    fn entries_per_cluster(&self) -> usize {
        return self.file_system.size_of_cluster() / Fat32FileSystem::size_of_directory_entry();
    }

    fn read_entry<const SIZE_OF_ENTRY: usize>(&mut self) -> Option<&[u8; SIZE_OF_ENTRY]> {
        let mut next_chunk = self.chunked_iter.next();
        if next_chunk.is_none() {
            let next_cluster = self.cluster_iter.next();
            let next_cluster = match next_cluster {
                None => return None,
                Some(cluster) => cluster.0,
            };

            self.file_system
                .read_cluster(next_cluster, &mut self.cluster_data);

            self.chunked_iter = self.cluster_data.chunks_exact(SIZE_OF_ENTRY);

            next_chunk = self.chunked_iter.next();
        }

        let chunk = match next_chunk {
            Some(chunk) => chunk,
            None => return None,
        };
        let chunk: Result<&[u8; SIZE_OF_ENTRY], _> = chunk.try_into();
        match chunk {
            Ok(chunk) => Some(chunk),
            Err(_) => None,
        }
    }

    fn parse_attributes_from_chunk<const SIZE_OF_ENTRY: usize>(
        entry_chunk: &[u8; SIZE_OF_ENTRY],
    ) -> Option<DirectoryAttributes> {
        let attribute_byte = entry_chunk.get(11..).and_then(|a| a.first());
        let attribute_byte = match attribute_byte {
            Some(attribute) => *attribute,
            None => return None,
        };

        let attribute: Result<DirectoryAttributes, _> = attribute_byte.try_into();

        match attribute {
            Ok(attribute) => Some(attribute),
            Err(_) => None,
        }
    }
}

impl<'a, 'b> Iterator for Fat32Entires<'a, 'b> {
    type Item = Fat32FileEntry;

    fn next(&mut self) -> Option<Self::Item> {
        const ENTRY_SIZE: usize = Fat32FileSystem::size_of_directory_entry();
        let entry_chunk = self.read_entry::<ENTRY_SIZE>();
        let entry_chunk = match entry_chunk {
            Some(entry_chunk) => entry_chunk,
            None => return None,
        };

        let attribute: Option<DirectoryAttributes> = Self::parse_attributes_from_chunk(entry_chunk);
        let attribute = match attribute {
            Some(attribute) => attribute,
            None => return None,
        };

        let mut directory_entry: DirectoryEntry;
        let long_file_name: Option<LongFileNameEntry>;

        if attribute.has_long_file_name() {
            long_file_name = Some(unsafe { core::mem::transmute(*entry_chunk) });

            let second_entry_chunk = self.read_entry::<ENTRY_SIZE>();
            let second_entry_chunk = match second_entry_chunk {
                Some(entry_chunk) => entry_chunk,
                None => return None,
            };

            directory_entry = unsafe { core::mem::transmute(*second_entry_chunk) };
        } else {
            long_file_name = None;
            directory_entry = unsafe { core::mem::transmute(*entry_chunk) };
        }

        Some(Fat32FileEntry {
            directory_entry: Some(directory_entry),
            long_file_name_entry: long_file_name,
        })
    }
}

pub struct Fat32File(Fat32FileEntry);

impl Fat32File {
    fn get_file_name(&self) -> Option<String> {
        self.0.get_file_name()
    }

    fn get_full_file_name(&self) -> Option<String> {
        let directory_entry = self.0.directory_entry;
        let directory_entry = match directory_entry {
            Some(directory_entry) => directory_entry,
            None => return None,
        };

        let file_name = self.get_file_name();
        let mut file_name = match file_name {
            Some(file_name) => file_name,
            None => return None,
        };

        let extension_empty = directory_entry.extension[0] != ascii::Char::Space.to_u8();
        if !extension_empty {
            file_name.push('.');

            append_ascii_to_string(
                &mut file_name,
                &directory_entry.extension,
                ascii::Char::Space,
            );
        }

        Some(file_name)
    }
}

impl TryFrom<Fat32FileEntry> for Fat32File {
    type Error = ();
    fn try_from(value: Fat32FileEntry) -> Result<Self, Self::Error> {
        let directory_entry = match value.directory_entry {
            Some(entry) => entry,
            None => return Err(()),
        };

        if directory_entry.attributes.is_file() {
            Ok(Fat32File(value))
        } else {
            Err(())
        }
    }
}

pub struct Fat32Directory(Fat32FileEntry);

impl Fat32Directory {
    fn get_file_name(&self) -> Option<String> {
        self.0.get_file_name()
    }

    fn root_directory() -> Self {
        return Self(Fat32FileEntry {
            long_file_name_entry: None,
            directory_entry: None,
        });
    }

    fn get_directory_cluster(&self, file_system: &Fat32FileSystem) -> DirectoryCluster {
        match self.0.directory_entry {
            Some(entry) => {
                let cluster = (entry.low_bits_of_cluster_number as usize)
                    | (entry.high_bits_of_cluster_number as usize) << 16;
                cluster.into()
            }
            None => (file_system.volume_header.root_directory_cluster as usize).into(),
        }
    }
}

impl TryFrom<Fat32FileEntry> for Fat32Directory {
    type Error = ();
    fn try_from(value: Fat32FileEntry) -> Result<Self, Self::Error> {
        match value.directory_entry {
            Some(entry) => {
                if entry.attributes.is_directory() {
                    Ok(Fat32Directory(value))
                } else {
                    Err(())
                }
            }
            None => Ok(Fat32Directory(value)),
        }
    }
}

impl Fat32FileSystem {
    #[inline]
    fn size_of_cluster(&self) -> usize {
        self.volume_header.sectors_per_cluster as usize
            * self.volume_header.bytes_per_sector as usize
    }

    #[inline]
    fn max_cluster(&self) -> usize {
        self.volume_header.large_sectors_per_fat as usize
            * self.volume_header.bytes_per_sector as usize
            / std::mem::size_of::<u32>()
    }

    #[inline]
    const fn min_cluster(&self) -> usize {
        2_usize
    }

    #[inline]
    const fn size_of_directory_entry() -> usize {
        sa::const_assert_eq!(
            core::mem::size_of::<DirectoryEntry>(),
            core::mem::size_of::<LongFileNameEntry>()
        );
        core::mem::size_of::<DirectoryEntry>()
    }

    fn get_cluster<'a, T>(&'a self, start_cluster: usize) -> Option<T>
    where
        T: Cluster,
    {
        if start_cluster >= self.min_cluster() && start_cluster <= self.max_cluster() {
            Some(T::from(start_cluster))
        } else {
            None
        }
    }

    fn get_clusters<'a, T>(&'a self, start_cluster: T) -> Option<ClustersIter<'a, T>>
    where
        T: Cluster,
    {
        Some(ClustersIter::new(self, start_cluster))
    }

    fn find_last_cluster<T>(&mut self, start_cluster: T) -> Option<usize>
    where
        T: Cluster,
    {
        let cluster = self.get_clusters(start_cluster).unwrap().last();

        match cluster {
            Some(cluster) => Some(cluster.into()),
            None => None,
        }
    }

    fn read_fat32_volume_header(
        disk: &mut Disk,
        partition_start_lba: usize,
        sector_size: usize,
    ) -> FAT32VolumeHeader {
        let count = usize::min(core::mem::size_of::<FAT32VolumeHeader>() / sector_size, 1);
        let mut bytes = vec![0_u8; count * sector_size];

        disk.read_sectors_from_disk(partition_start_lba, count, sector_size, &mut bytes);
        bytes.resize(core::mem::size_of::<FAT32VolumeHeader>(), 0_u8);

        let bytes: [u8; core::mem::size_of::<FAT32VolumeHeader>()] = bytes.try_into().unwrap();
        let mut volume_header: FAT32VolumeHeader = unsafe { mem::transmute(bytes) };

        volume_header.beginning_lba = partition_start_lba as u32;
        volume_header.bytes_per_sector = sector_size as u16;

        volume_header
    }

    fn write_fat32_volume_header(&mut self) {
        let partition_start_lba = self.volume_header.beginning_lba as usize;
        let bytes: [u8; core::mem::size_of::<FAT32VolumeHeader>()] =
            unsafe { mem::transmute(self.volume_header) };

        let count = usize::min(
            bytes.len() / self.volume_header.bytes_per_sector as usize,
            1,
        );

        self.disk.write_sectors_to_disk(
            partition_start_lba,
            count,
            self.volume_header.bytes_per_sector as usize,
            &bytes,
        );
    }

    fn read_fs_info(disk: &mut Disk, volume_header: &FAT32VolumeHeader) -> FSInfo {
        let mut bytes = [0_u8; std::mem::size_of::<FSInfo>()];
        disk.read_sectors_from_disk(
            volume_header.beginning_lba as usize
                + volume_header.location_of_fs_info_sector as usize,
            1,
            volume_header.bytes_per_sector as usize,
            &mut bytes,
        );
        unsafe { mem::transmute(bytes) }
    }

    fn update_fs_info(&mut self) {
        let should_update_cluster_count = self.fs_info.last_known_free_cluster_count == 0xFFFFFFFF;

        if !should_update_cluster_count && self.fs_info.available_clusters_start_index != 0xFFFFFFFF
        {
            return;
        }

        let mut first_avaliable_cluster = 0_usize;
        let mut avaliable_cluster_count = 0_usize;

        for cluster in self.fat.iter() {
            if *cluster != 0 {
                continue;
            }

            if first_avaliable_cluster == 0 {
                first_avaliable_cluster = *cluster as usize;
            }

            if !should_update_cluster_count {
                break;
            }

            avaliable_cluster_count += 1;
        }

        self.fs_info.available_clusters_start_index = first_avaliable_cluster as u32;

        if should_update_cluster_count {
            self.fs_info.last_known_free_cluster_count = avaliable_cluster_count as u32;
        }
    }

    fn read_fat(disk: &mut Disk, volume_header: &FAT32VolumeHeader) -> Vec<u32> {
        let fat_begin_lba =
            volume_header.beginning_lba as usize + volume_header.reserved_sectors as usize;
        let fat_sector_count =
            volume_header.fat_count as usize * volume_header.large_sectors_per_fat as usize;

        let mut buffer = vec![0_u8; fat_sector_count * volume_header.bytes_per_sector as usize];
        disk.read_sectors_from_disk(
            fat_begin_lba,
            fat_sector_count,
            volume_header.bytes_per_sector as usize,
            &mut buffer,
        );

        let fat_size = volume_header.large_sectors_per_fat as usize
            * volume_header.bytes_per_sector as usize
            / 4;

        let mut fats: Vec<u32> = vec![0_u32; fat_size];
        for (i, buffer_chunk) in buffer.chunks_exact(4).enumerate() {
            let fat_index = i / fat_size;
            let index = i % fat_size;

            let cluster_chunk: Result<[u8; 4], _> = buffer_chunk.try_into();
            let cluster = match cluster_chunk {
                Ok(cluster) => cluster,
                Err(_) => panic!("Chunk to sized array of 4 failed"),
            };
            let cluster = u32::from_le_bytes(cluster);

            if fat_index > 0 {
                fats[index] = cluster;
            } else if fats[index] != cluster {
                // What should we do? Probably just ignore it ig
                // panic!("Invalid fat chunk found!");
            }
        }

        fats
    }

    fn write_fat(&mut self) {
        let fat_begin_lba = self.volume_header.beginning_lba as usize
            + self.volume_header.reserved_sectors as usize;
        let fat_sector_count = self.volume_header.fat_count as usize
            * self.volume_header.large_sectors_per_fat as usize;

        let mut bytes = vec![0_u8; self.fat.len() * std::mem::size_of::<u32>()];
        for (i, fat_cluster_num) in self.fat.iter().enumerate() {
            let bytes_slice =
                &mut bytes[i * std::mem::size_of::<u32>()..][..std::mem::size_of::<u32>()];
            bytes_slice.copy_from_slice(&fat_cluster_num.to_le_bytes());
        }

        if self.volume_header.fat_count > 1 {
            let bytes_clone = bytes.clone();
            for _ in 1..self.volume_header.fat_count {
                bytes.extend(bytes_clone.iter());
            }
        }

        // Pad to sector size (Should I do this?)
        // let byte_padding_size = volume_header.bytes_per_sector as usize
        //     - (bytes.len() % volume_header.bytes_per_sector as usize) as usize;
        // bytes.extend(vec![0_u8; byte_padding_size]);

        self.disk.write_sectors_to_disk(
            fat_begin_lba,
            fat_sector_count,
            self.volume_header.bytes_per_sector as usize,
            &bytes,
        );
    }

    fn read_cluster(&mut self, cluster_index: usize, buffer: &mut [u8]) {
        let cluster_begin_lba = self.volume_header.beginning_lba as usize
            + self.volume_header.reserved_sectors as usize
            + (self.volume_header.fat_count as usize
                * self.volume_header.large_sectors_per_fat as usize);
        let cluster_size_in_sectors = self.volume_header.sectors_per_cluster as usize;

        self.disk.read_sectors_from_disk(
            cluster_begin_lba + (cluster_index - 2),
            cluster_size_in_sectors,
            self.volume_header.bytes_per_sector as usize,
            buffer,
        );
    }

    fn write_cluster(&mut self, cluster_index: usize, buffer: &[u8]) {
        let cluster_begin_lba = self.volume_header.beginning_lba as usize
            + self.volume_header.reserved_sectors as usize
            + (self.volume_header.fat_count as usize
                * self.volume_header.large_sectors_per_fat as usize);
        let cluster_size_in_sectors = self.volume_header.sectors_per_cluster as usize;

        self.disk.write_sectors_to_disk(
            cluster_begin_lba + (cluster_index - 2),
            cluster_size_in_sectors,
            self.volume_header.bytes_per_sector as usize,
            buffer,
        );
    }

    fn get_avaliable_cluster_count(&mut self) -> usize {
        self.update_fs_info();
        self.fs_info.last_known_free_cluster_count as usize
    }

    fn link_clusters<const CLUSTER_COUNT: usize>(&mut self, clusters: [u32; CLUSTER_COUNT]) {
        for (prev_cluster, cur_cluster) in clusters.iter().zip(clusters.iter().skip(1)) {
            // If we are updating a cluster chain we need to update fs_info
            if self.fs_info.available_clusters_start_index == *prev_cluster {
                self.fs_info.available_clusters_start_index = 0xFFFFFFFF;
            }

            println!("Linking {:x}->{:x}", *prev_cluster, *cur_cluster);
            self.fat[*prev_cluster as usize] = *cur_cluster;
            self.fat[*cur_cluster as usize] = 0xFFFFFFFF;
        }
        self.write_fat();
    }

    fn get_avaliable_clusters<const CLUSTER_COUNT: usize>(
        &mut self,
    ) -> Option<[u32; CLUSTER_COUNT]> {
        if self.get_avaliable_cluster_count() < CLUSTER_COUNT || CLUSTER_COUNT == 0 {
            return None;
        }

        let mut result_avaliable_clusters = [0_u32; CLUSTER_COUNT];
        let mut avaliable_cluster_index = 0;

        let mut fat_table_index = self.fs_info.available_clusters_start_index as usize;

        while avaliable_cluster_index < CLUSTER_COUNT {
            if fat_table_index >= self.fat.len() {
                return None;
            }

            let cluster = self.fat[fat_table_index];
            if cluster == 0 {
                result_avaliable_clusters[avaliable_cluster_index] = fat_table_index as u32;
                avaliable_cluster_index += 1;
            }
            fat_table_index += 1;
        }

        Some(result_avaliable_clusters)
    }

    fn get_directory_from_path(&mut self, path: &str) -> Result<Fat32Directory, FileSystemError> {
        let path_directories: Vec<&str> = match path.trim().len() {
            0 => {
                return Ok(Fat32Directory::root_directory());
            }
            _ => path.split('\\').collect(),
        };

        for path in path_directories.iter() {
            if path.trim().is_empty() {
                return Err(FileSystemError(String::from("Invalid Path!")));
            }
        }

        let mut search_path_index = 0_usize;
        let mut search_directory = Fat32Directory::root_directory();
        let mut file_entries =
            Fat32Entires::new(self, search_directory.get_directory_cluster(self));

        while search_path_index < path_directories.len() {
            let file_entry = file_entries.next();
            match file_entry {
                Some(file) => {
                    let file_name = file.get_file_name();
                    let file_name = match file_name {
                        Some(file_name) => file_name,
                        None => continue,
                    };

                    let directory: Result<Fat32Directory, _> = file.try_into();
                    if let Ok(directory) = directory {
                        let names_equal = file_name.to_lowercase()
                            == path_directories[search_path_index].to_lowercase();
                        if names_equal {
                            file_entries =
                                Fat32Entires::new(self, directory.get_directory_cluster(self));
                            search_directory = directory;
                            search_path_index += 1;
                        }
                    }
                }
                None => break,
            }
        }

        if search_path_index == path_directories.len() {
            return Ok(search_directory);
        }

        Err(FileSystemError(String::from("Couldn't find directory!")))
    }

    fn parse_file_name(full_file_name: &str) -> (&str, &str) {
        let file_name: &str;
        let extension: &str;
        let file_slice = full_file_name.rfind('.');
        match file_slice {
            Some(index) => {
                file_name = &full_file_name.get(..index).unwrap();
                extension = &full_file_name.get((index + 1)..).unwrap_or_default();
            }
            None => {
                file_name = full_file_name;
                extension = "";
            }
        }
        (file_name, extension)
    }
}

impl filesystem::FileSystem<Fat32File, Fat32Directory> for Fat32FileSystem {
    fn create_from_disk(mut disk: Disk, partition_start_lba: usize) -> Self {
        let sector_size = 512_usize; // We would read this from the hardware

        let volume_header =
            Self::read_fat32_volume_header(&mut disk, partition_start_lba, sector_size);
        let fs_info = Self::read_fs_info(&mut disk, &volume_header);
        let fat = Self::read_fat(&mut disk, &volume_header);

        let mut file_system = Fat32FileSystem {
            disk,
            volume_header,
            fs_info,
            fat,
        };
        file_system.write_fat32_volume_header();

        file_system
    }

    fn load_file(
        &mut self,
        full_file_name: &str,
        path: &str,
    ) -> Result<Fat32File, filesystem::FileSystemError> {
        let directory = self.get_directory_from_path(path);
        let directory = match directory {
            Ok(a) => a,
            Err(err) => {
                return Err(err);
            }
        };

        let directory_cluster = directory.get_directory_cluster(self);
        let file_entries = Fat32Entires::new(self, directory_cluster);

        for file_entry in file_entries {
            let file_entry: Result<Fat32File, _> = file_entry.try_into();
            let file_entry = match file_entry {
                Ok(entry) => entry,
                Err(_) => continue,
            };

            let search_full_file_name = match file_entry.get_full_file_name() {
                Some(str) => str,
                None => continue,
            };

            let names_equal = search_full_file_name.to_lowercase() == full_file_name.to_lowercase();
            if names_equal {
                return Ok(file_entry);
            }
        }

        Err(FileSystemError(String::from("Couldn't find file!")))
    }

    fn create_file(
        &mut self,
        full_file_name: &str,
        path: &str,
        file_attributes: filesystem::FileAttributes,
    ) -> Result<Fat32File, filesystem::FileSystemError> {
        // let directory = self.get_directory_from_path(path);
        // let directory = match directory {
        //     Ok(a) => a,
        //     Err(err) => {
        //         return Err(err);
        //     }
        // };

        // let (file_name, extension) = Self::parse_file_name(full_file_name);

        // let mut cluster = directory.get_data_cluster(&self.volume_header);
        // let mut cluster_buffer = vec![0_u8; self.size_of_cluster()];

        // self.read_cluster(cluster, &mut cluster_buffer);

        // let mut chunk_iter = cluster_buffer
        //     .chunks_exact_mut(Self::size_of_directory_entry())
        //     .enumerate();

        // // TODO: Fix this "cursed" chunked iteration alloc loop thingy
        // let mut clusters_allocated = 0_usize;
        // let fat_file: Option<Fat32File>;

        // loop {
        //     let chunk = chunk_iter.next();

        //     let (i, chunk) = match chunk {
        //         Some(chunk) => chunk,
        //         None => {
        //             if clusters_allocated > 0 {
        //                 // Some thing is going really wrong. Allocating way too many clusters for a directory entry.
        //                 panic!("Allocating too many clusters!");
        //             }

        //             // We need to allocate another cluster to put the next directory entry
        //             let prev_cluster = cluster;

        //             let clusters = self.get_avaliable_clusters::<1>();
        //             cluster = match clusters {
        //                 Some(file) => file[0] as usize,
        //                 None => return Err(FileSystemError(String::from("Disk is full!"))),
        //             };
        //             println!("Next cluster: {}", cluster);

        //             self.link_clusters([prev_cluster as u32, cluster as u32]);
        //             self.read_cluster(cluster, &mut cluster_buffer);

        //             chunk_iter = cluster_buffer
        //                 .chunks_exact_mut(Self::size_of_directory_entry())
        //                 .enumerate();
        //             clusters_allocated += 1;
        //             continue;
        //         }
        //     };

        //     // Hint to the compiler that we know the size of the array so we can transmute
        //     let sized_chunk: Result<&mut [u8; Self::size_of_directory_entry()], _> =
        //         chunk.try_into();

        //     let chunk = match sized_chunk {
        //         Ok(sized_chunk) => sized_chunk,
        //         Err(_) => return Err(FileSystemError(String::from("Error"))),
        //     };

        //     let all_zeros = chunk.iter().all(|&x| x == 0);
        //     if all_zeros {
        //         println!("Creating file at {}", i);

        //         let mut attributes = DirectoryAttributes(DirectoryAttributes::FILE);
        //         if file_attributes.is_read_only() {
        //             println!("Read Only!");
        //             attributes.0 |= DirectoryAttributes::READ_ONLY;
        //         }
        //         if file_attributes.is_hidden() {
        //             println!("Hidden!");
        //             attributes.0 |= DirectoryAttributes::HIDDEN;
        //         }

        //         let directory_entry =
        //             DirectoryEntry::new(file_name, extension, 0xFFFFFFFF, 0x0, attributes);

        //         fat_file = Some(Fat32File {
        //             directory_entry,
        //             optional_lfn: None,
        //         });

        //         let directory_entry: [u8; Self::size_of_directory_entry()] =
        //             unsafe { core::mem::transmute(directory_entry) };
        //         println!("{:?}", directory_entry);

        //         chunk.copy_from_slice(&directory_entry);
        //         break;
        //     } else {
        //     }
        // }

        // match fat_file {
        //     Some(file) => {
        //         self.write_cluster(cluster, &cluster_buffer);
        //         Ok(file)
        //     }
        //     None => Err(FileSystemError(String::from("Unable to create file!"))),
        // }
        Err(FileSystemError(String::from("No Impl!")))
    }

    fn create_directory(
        &mut self,
        directory_name: &str,
        path: &str,
        attributes: filesystem::DirectoryAttributes,
    ) -> Result<Fat32Directory, filesystem::FileSystemError> {
        Err(FileSystemError(String::from("No Impl!")))
    }

    fn write_to_file(
        &mut self,
        file: Fat32File,
        bytes: &[u8],
        write_mode: filesystem::WriteMode,
    ) -> Result<filesystem::WriteResult, filesystem::FileSystemError> {
        Err(FileSystemError(String::from("No Impl!")))
    }

    fn read_file(
        &mut self,
        file: Fat32File,
        number_of_bytes: usize,
    ) -> Result<filesystem::ReadResult, filesystem::FileSystemError> {
        Err(FileSystemError(String::from("No Impl!")))
    }
}
