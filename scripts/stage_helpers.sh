# Shared by stages 7-9 (eval'ed after their settle/drop/run/capfor).

cal(){  # cal <out.json> <PHASOR peak GiB> <budget> <command with {B} {OUT}>: memcal.py, redone if one
  # of its probes was stopped because memory was held outside the run (in_cgroup.sh HOSTPRESSURE):
  # such a probe would otherwise count as a setting that does not fit
  local c=$1 T=$2 b=$3 n; shift 3
  for n in 1 2 3; do
    if [ -s $c ]; then
      grep -q HOSTPRESSURE $c.log 2>/dev/null && python3 -c "import json,sys;sys.exit(0 if json.load(open('$c')).get('killed') else 1)" || return 0
      [ $n = 3 ] && return 0
      mv -f $c.log $c.log.pressure; rm -f $c
    fi
    settle; drop; timeout 14400 python3 scripts/memcal.py $T $b $c -- "$@" > $c.log 2>&1
  done
}

flag14(){  # flag14 <out> <PHASOR peak GiB> <budget>: a run whose peak exceeds 1.4 x PHASOR's is recorded as not fitting
  local o=$1 T=$2 nb=$3 pk
  grep -q '^RESULT' $o.txt 2>/dev/null || return 0
  pk=$(grep '^RESULT' $o.txt | grep -o ' peak_gib=[0-9.]*' | head -1 | grep -o '[0-9.]*')
  [ -n "$pk" ] && awk -v p=$pk -v t=$T 'BEGIN{exit !(p>1.4*t)}' && echo "NORUN budget=$nb reason=exceeds-1.4x-phasor-peak (peak $pk vs $T)" >> $o.txt
  return 0
}

over14(){  # over14 <out> <PHASOR peak GiB> <budget>: the run at the smallest cache (1 GiB) for a budget whose
  # memcal found no setting within PHASOR's peak; it counts only within 1.4 x that peak
  local o=$1 T=$2 nb=$3
  echo "NOTE smallest-cache-under-1.4x-phasor-peak (no setting within PHASOR's peak $T GiB)" >> $o.txt
  if grep -q '^RESULT' $o.txt; then flag14 $o $T $nb
  elif grep -q '^NORUN' $o.txt; then sed -i 's/reason=oom-under-cap/reason=exceeds-1.4x-phasor-peak/' $o.txt; fi
  return 0
}

because(){  # because <out> <budget>: a run that ended without a result or a NORUN line gets its cause recorded
  local o=$1 nb=$2
  [ -f $o.txt ] || return 0
  grep -qE '^(RESULT|NORUN)' $o.txt && return 0
  echo "NORUN budget=$nb reason=failed: $(grep -hE 'Error|error|HOSTGUARD|CAPTHRASH|Killed' $o.txt | tail -1 | tr ' ' '_' | cut -c1-160)" >> $o.txt
}
