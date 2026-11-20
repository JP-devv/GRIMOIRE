# Unprofiled arms must have exactly one aggregate measurement and no traces.
/^Verify attention:|^QFN verifier KV:|^QFN MTP3 verify head:|^  qwen4_exp prefill M=.*\(sub-chunks|fenced\)/ { bad=1 }
/^generation phases:/ {
    if (NF!=6 || $3!="prefill_tokens=2706" || $5!="decode_tokens=31" ||
        $4!~/^prefill_ms=[0-9]+\.[0-9]+$/ || $6!~/^decode_ms=[0-9]+\.[0-9]+$/) bad=1
    pp=substr($4,12)+0; tg=substr($6,11)+0
    ++n
}
END {
    if (bad || n!=1 || pp<=0 || tg<=0) exit 1
    printf "%.3f\t%.3f\n", pp, tg
}
