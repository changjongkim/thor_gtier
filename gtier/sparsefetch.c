// The granularity floor, formalised.
//
// A sparse workload needs a set of items of size g scattered through a file.
// You can fetch each item exactly (no waste, but small I/O is slow), or fetch a
// larger aligned block around it (fast I/O, but you pay for bytes you discard).
// Useful bandwidth is what the application actually gets:
//
//     useful = (items * g) / elapsed
//
// Sweeping (g, block) gives the whole trade-off surface and answers two things:
// the best staging size for a given sparsity granularity, and the granularity
// below which no choice is good enough -- the floor.
#define _GNU_SOURCE
#include <fcntl.h>
#include <liburing.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static double now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
  return t.tv_sec + t.tv_nsec*1e-9; }

static unsigned long long rnd(unsigned long long *s){
  *s ^= *s<<13; *s ^= *s>>7; *s ^= *s<<17; return *s; }

int main(int argc, char **argv){
  const char *path = argv[1];
  size_t item   = strtoull(argv[2],0,10);   // sparsity granularity g, bytes
  size_t block  = strtoull(argv[3],0,10);   // aligned fetch size, bytes
  size_t useful = strtoull(argv[4],0,10)<<20; // useful bytes to deliver, MiB
  int depth     = argc>5 ? atoi(argv[5]) : 64;

  if (block < item) block = item;
  int fd = open(path, O_RDONLY|O_DIRECT);
  if(fd<0){ perror("open"); return 1; }
  off_t sz = lseek(fd,0,SEEK_END);
  size_t nblk = sz/block;
  size_t nitems = useful/item;

  void **bufs = malloc(sizeof(void*)*depth);
  for(int i=0;i<depth;i++) if(posix_memalign(&bufs[i],4096,block)) return 1;

  struct io_uring ring;
  if(io_uring_queue_init(depth,&ring,0)<0){ perror("uring"); return 1; }

  unsigned long long seed = 0x9E3779B97F4A7C15ull;
  size_t issued=0, done=0;
  double t0=now();
  for(int i=0;i<depth && issued<nitems;i++,issued++){
    struct io_uring_sqe *s=io_uring_get_sqe(&ring);
    io_uring_prep_read(s,fd,bufs[i],block,(off_t)(rnd(&seed)%nblk)*block);
    io_uring_sqe_set_data64(s,i);
  }
  io_uring_submit(&ring);
  while(done<nitems){
    struct io_uring_cqe *c;
    if(io_uring_wait_cqe(&ring,&c)<0) return 1;
    int slot=(int)io_uring_cqe_get_data64(c);
    if(c->res<0){ fprintf(stderr,"read %s\n",strerror(-c->res)); return 1; }
    io_uring_cqe_seen(&ring,c); done++;
    if(issued<nitems){
      struct io_uring_sqe *s=io_uring_get_sqe(&ring);
      io_uring_prep_read(s,fd,bufs[slot],block,(off_t)(rnd(&seed)%nblk)*block);
      io_uring_sqe_set_data64(s,slot);
      io_uring_submit(&ring); issued++;
    }
  }
  double t=now()-t0;
  double raw    = (double)nitems*block/(1ull<<30)/t;
  double usefulbw = (double)nitems*item /(1ull<<30)/t;
  printf("item=%7zuB block=%8zuB amp=%6.1fx  raw=%6.3f GiB/s  useful=%6.3f GiB/s\n",
         item, block, (double)block/item, raw, usefulbw);
  io_uring_queue_exit(&ring); close(fd); return 0;
}
