CC = gcc
CFLAGS = -fPIC -shared -Icspice/include

project.so: project.c
	$(CC) $(CFLAGS) -o project.so project.c cspice/lib/cspice.a -lm

clean:
	rm -rf project.so
