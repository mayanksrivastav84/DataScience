**Remove header from all the files in UNIX**

    cd path-to-airline-data
    for file_name in *csv;
    do
    bunzip2 $file_name
    base_name=$(echo $file_name | cut -d'.' -f1,2);
    awk '{if (NR!=1) {print}}' $base_name > nh_${base_name}
    hadoop fs -put nh_${base_name} /airline/
    echo $base_name
    done
