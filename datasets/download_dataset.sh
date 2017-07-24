# celeba dataset
download_celeba(){
    echo "----------------------- downloading celeba dataset -----------------------"
    wget https://storage.googleapis.com/stext2image/face_pretrained.tar.gz
    tar xzvf celeba.tar.gz
}

download_all(){
    download_celeba
    echo "----------------------- done -----------------------"
}

download_all
