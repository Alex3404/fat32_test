use crate::disk::Disk;

pub struct FileAttributes(u8);
impl FileAttributes {
    pub const NONE: Self = Self(0x00);
    pub const HIDDEN: Self = Self(0x01);
    pub const READ_ONLY: Self = Self(0x02);

    pub fn is_hidden(&self) -> bool {
        self.0 & Self::HIDDEN.0 == Self::HIDDEN.0
    }

    pub fn is_read_only(&self) -> bool {
        self.0 & Self::READ_ONLY.0 == Self::READ_ONLY.0
    }
}

pub struct DirectoryAttributes(u8);
impl DirectoryAttributes {
    pub const NONE: Self = Self(0x00);
    pub const HIDDEN: Self = Self(0x01);
    pub const READ_ONLY: Self = Self(0x02);

    pub fn is_hidden(&self) -> bool {
        self.0 & Self::HIDDEN.0 == Self::HIDDEN.0
    }

    pub fn is_read_only(&self) -> bool {
        self.0 & Self::READ_ONLY.0 == Self::READ_ONLY.0
    }
}

pub struct FileSystemError(pub String);

pub struct WriteResult {}
pub struct WriteMode(pub u8);
impl WriteMode {
    pub const OVERWRITE: Self = Self(0x01);
    pub const APPEND: Self = Self(0x02);
}

pub struct ReadResult(pub Vec<u8>);

pub trait FileSystem<FileT, DirectoryT> {
    fn create_from_disk(disk: Disk, partition_start_lba: usize) -> Self;

    fn create_file(
        &mut self,
        file_name: &str,
        path: &str,
        attributes: FileAttributes,
    ) -> Result<FileT, FileSystemError>;
    fn load_file(&mut self, file_name: &str, path: &str) -> Result<FileT, FileSystemError>;
    fn create_directory(
        &mut self,
        directory_name: &str,
        path: &str,
        attributes: DirectoryAttributes,
    ) -> Result<DirectoryT, FileSystemError>;

    fn write_to_file(
        &mut self,
        file: FileT,
        bytes: &[u8],
        write_mode: WriteMode,
    ) -> Result<WriteResult, FileSystemError>;
    fn read_file(
        &mut self,
        file: FileT,
        number_of_bytes: usize,
    ) -> Result<ReadResult, FileSystemError>;
}
