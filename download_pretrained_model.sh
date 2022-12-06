# face model
download_face(){
    echo "----------------------- downloading face pretrained model -----------------------"
    wget -O face_pretrained.tar.gz "https://www.robots.ox.ac.uk/~szwu/storage/18_sketch/face_pretrained.tar.gz"
    tar xzvf face_pretrained.tar.gz
    rm face_pretrained.tar.gz
}

# all models
download_all(){
    download_face
    echo "----------------------- done -----------------------"
}

download_all
