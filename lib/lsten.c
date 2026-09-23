#include "gguf.h"
#include <stdio.h>
#include <string.h>
int main(int argc,char**argv){
  unsigned long long ffn=0,att=0,other=0; int nexp=0;
  for(int a=1;a<argc;a++){
    gguf_model m; if(gguf_load(argv[a],&m)) continue;
    for(int i=0;i<m.n;i++){
      const char*n=m.t[i].name;
      if(strstr(n,"_exps")){ ffn+=m.t[i].size; nexp++;
        if(nexp<=3) printf("  expert tensor: %-38s %8.2f MiB\n",n,m.t[i].size/1048576.0); }
      else if(strstr(n,"attn")) att+=m.t[i].size;
      else other+=m.t[i].size;
    }
    gguf_free(&m);
  }
  double tot=(ffn+att+other)/1073741824.0;
  printf("\n  expert(ffn_*_exps) %7.2f GiB  (%.1f%%)  텐서 %d개\n",ffn/1073741824.0,100.0*ffn/(ffn+att+other),nexp);
  printf("  attention          %7.2f GiB  (%.1f%%)\n",att/1073741824.0,100.0*att/(ffn+att+other));
  printf("  기타               %7.2f GiB  (%.1f%%)\n",other/1073741824.0,100.0*other/(ffn+att+other));
  printf("  합계               %7.2f GiB\n",tot);
  return 0;}
