# face model
download_face(){
    echo "----------------------- downloading face pretrained model -----------------------"
    wget -O face_pretrained.tar.gz "https://unioxfordnexus-my.sharepoint.com/:u:/g/personal/sann6319_ox_ac_uk/EUkbL8V7VMxFlLad2R2cJxMBtcA0Im4bZJ-pK_izyiQuHA?e=GYcpPf&download=1"
    tar xzvf face_pretrained.tar.gz
    rm face_pretrained.tar.gz
}

# all models
download_all(){
    download_face
    echo "----------------------- done -----------------------"
}

download_all
