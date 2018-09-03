# celeba dataset
download_celeba(){
    echo "----------------------- downloading celeba dataset -----------------------"
    wget -O celeba.tar.gz "https://unioxfordnexus-my.sharepoint.com/:u:/g/personal/sann6319_ox_ac_uk/ES8bgYAGwXBDnpAA-4q99qgBzXziqP5seSUjdFsMvHcXzg?e=CuI830&download=1"
    tar xzvf celeba.tar.gz -C ./datasets/
    rm celeba.tar.gz
}

# all datasets
download_all(){
    download_celeba
    echo "----------------------- done -----------------------"
}

download_all
