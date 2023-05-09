FROM ubuntu
LABEL maintainer=goncharov.mv@phystech.edu
RUN apt-get update && apt-get install -y vim curl git && apt-get clean

RUN git clone -c http.extraHeader='Authorization: Bearer BBDC-ODkxMTI2ODUxODAxOpjyYYq9A1Nid/K1KmLnhv/P+SUf' https://workshop.samcs.ru/bitbucket/scm/goab/go-broadcast.git

WORKDIR go-broadcast

RUN apt-get install python3 pip -y

RUN pip3 install torch==2.0.0+cpu torchvision==0.15.1+cpu --index-url https://download.pytorch.org/whl/cpu
RUN pip3 install -r requirements.txt
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get install python3-tk -y

RUN apt-get install libqt5gui5 -y
RUN apt-get install libpulse-mainloop-glib0 -y

RUN pip3 install gdown
RUN gdown "https://drive.google.com/uc?export=download&id=1LyMUY7dfIr-Uf3Elw-0tgwtF3fehqCDb" -O src/state_recognition/model_saves/yolo8n_608_1200.pt
RUN gdown "https://drive.google.com/uc?export=download&id=1_bQJXK3Qaxg6yO9NR6VgIHT2AlgPaoBS" -O src/state_recognition/model_saves/segmentation18.pth

# run curl -L https://bootstrap.saltstack.com -o install_salt.sh && bash install_salt.sh -P -M
# ADD post_configure.sh /go-broadcast/post_configure.sh
# RUN chmod +x /go-broadcast/post_configure.sh
# CMD /go-broadcast/post_configure.sh
