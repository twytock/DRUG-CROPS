declare -a dataset_arr 
while IFS=$',' read -r -a myArray
do
	dataset_arr=("${dataset_arr[@]}" "${myArray[0]}")
done < datasets2.txt
tLen=${#dataset_arr[@]} 
for ((i=0; i<${tLen}; i++));
do
	python -m scoop -n 8 mrp_miqp_alt_obj_evk.py -d ${dataset_arr[i]} -n 37 -f drug_feat_altobj_l -N 2 -C $1 -V "_1121" -Z _0122 &
done
