#include "gguf.h"
#include <string.h>
#include <stdio.h>
int main(int argc,char**argv){
  for(int a=1;a<argc;a++){
    gguf_model m; if(gguf_load(argv[a],&m)){printf("%s PARSE FAIL\n",argv[a]);continue;}
    unsigned long long mx=0; const char*nm="";
    for(int i=0;i<m.n;i++){unsigned long long e=m.t[i].offset+m.t[i].size; if(e>mx){mx=e;nm=m.t[i].name;}}
    printf("%-58s file=%12llu maxextent=%12llu %s %s\n",
      strrchr(argv[a],'/')+1,(unsigned long long)m.file_bytes,mx,
      mx>m.file_bytes?"OVERRUN":"ok",mx>m.file_bytes?nm:"");
    gguf_free(&m);
  }
  return 0;}
