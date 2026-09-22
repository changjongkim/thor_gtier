// Sequential vs scattered reads on this NVMe.  A MoE model reads ~9% of its
// weights per token, scattered by expert routing; if scattered reads are much
// slower than sequential ones, storage layout -- not the paging mechanism --
// is what decides whether out-of-core MoE inference is viable.
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

int main(int argc, char **argv){
  const char *path = argv[1];
  size_t blk = argc>2 ? strtoull(argv[2],0,10) : 65536;   // read size
  int rnd     = argc>3 ? atoi(argv[3]) : 1;               // 1=random 0=seq
  int depth   = argc>4 ? atoi(argv[4]) : 64;
  size_t total= argc>5 ? strtoull(argv[5],0,10)<<20 : (2ull<<30);

  int fd = open(path, O_RDONLY|O_DIRECT);
  if(fd<0){ perror("open"); return 1; }
  off_t sz = lseek(fd,0,SEEK_END);
  size_t nblk = sz/blk, nread = total/blk;

  void **bufs = malloc(sizeof(void*)*depth);
  for(int i=0;i<depth;i++) if(posix_memalign(&bufs[i],4096,blk)){ perror("memalign"); return 1; }

  struct io_uring ring;
  if(io_uring_queue_init(depth,&ring,0)<0){ perror("uring"); return 1; }

  unsigned long long seed = 88172645463325252ull;
  size_t issued=0, done=0;
  double t0=now();
  for(int i=0;i<depth && issued<nread;i++,issued++){
    size_t b = rnd ? ({ seed^=seed<<13; seed^=seed>>7; seed^=seed<<17; seed%nblk; }) : issued%nblk;
    struct io_uring_sqe *s=io_uring_get_sqe(&ring);
    io_uring_prep_read(s,fd,bufs[i],blk,(off_t)b*blk);
    io_uring_sqe_set_data64(s,i);
  }
  io_uring_submit(&ring);
  while(done<nread){
    struct io_uring_cqe *c;
    if(io_uring_wait_cqe(&ring,&c)<0){ perror("cqe"); return 1; }
    int slot=(int)io_uring_cqe_get_data64(c);
    if(c->res<0){ fprintf(stderr,"read %s\n",strerror(-c->res)); return 1; }
    io_uring_cqe_seen(&ring,c); done++;
    if(issued<nread){
      size_t b = rnd ? ({ seed^=seed<<13; seed^=seed>>7; seed^=seed<<17; seed%nblk; }) : issued%nblk;
      struct io_uring_sqe *s=io_uring_get_sqe(&ring);
      io_uring_prep_read(s,fd,bufs[slot],blk,(off_t)b*blk);
      io_uring_sqe_set_data64(s,slot);
      io_uring_submit(&ring); issued++;
    }
  }
  double t=now()-t0;
  printf("%-10s blk=%6zuKiB depth=%3d : %7.3f GiB/s  (%6.0f kIOPS)\n",
         rnd?"random":"sequential", blk>>10, depth,
         (double)total/(1ull<<30)/t, nread/t/1000.0);
  io_uring_queue_exit(&ring); close(fd); return 0;
}
