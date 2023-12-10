for path in "/src/dotgraphs/"*.dot
do
f=${path##*/};
dot -Txdot_json "$path" >  "/dst/jsongraphs/${f%.*}.json";
done

