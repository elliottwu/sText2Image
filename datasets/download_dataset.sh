# celeba dataset
download_celeba(){
    echo "----------------------- downloading celeba dataset -----------------------"
    wget -O celeba.tar.gz "https://www.robots.ox.ac.uk/~szwu/storage/18_sketch/celeba.tar.gz"
    tar xzvf celeba.tar.gz -C ./datasets/
    rm celeba.tar.gz
}

# all datasets
download_all(){
    download_celeba
    echo "----------------------- done -----------------------"
}

download_all
