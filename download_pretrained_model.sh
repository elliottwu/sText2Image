# face model
download_face(){
    echo "----------------------- downloading face pretrained_model -----------------------"
    wget https://storage.googleapis.com/stext2image/face_pretrained.tar.gz
    tar xzvf face_pretrained.tar.gz
}

# all models
download_all(){
    download_face
    echo "----------------------- done -----------------------"
}

download_all
