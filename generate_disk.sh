dd if=/dev/zero of=disk.img bs=1M count=2
mformat -i disk.img :: -F
mcopy -i disk.img hello.txt ::
mcopy -i disk.img longfile.txt ::
mcopy -i disk.img folder ::