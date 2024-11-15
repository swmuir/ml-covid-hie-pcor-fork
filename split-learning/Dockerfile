#################################CPU############################################
ARG BASE_IMAGE=python:3.11-bookworm
FROM ${BASE_IMAGE} AS base
COPY requirements.txt requirements.txt
RUN pip3 install --upgrade pip
RUN pip install -r requirements.txt && pip install pylint

# conditionally run apt based on the os-release ID
RUN . /etc/os-release &&\
    if [ "${ID}" != "amzn" ] ;  then \ 
    apt update && apt install openvpn iputils-ping net-tools netcat-openbsd tcpdump iproute2 psmisc vim iptables telnet tcpdump stunnel supervisor gettext wget -y && \
    sed -i 's/#net.ipv4.ip_forward=1/net.ipv4.ip_forward=1/g' /etc/sysctl.conf  && \
    echo 'net.ipv6.conf.all.disable_ipv6=0' >> /etc/sysctl.conf && \
    echo 'tcpdump -i any  -s 65535 -vvv -w /tcpdump-logs/tcpdump_raw.dmp --print > /tcpdump-logs/tcpdump_print.log 2>&1 &' >> /tmp/init.sh ;\
    fi

COPY logging/prefix-log.sh /tmp/prefix-log.sh
RUN chmod +x /tmp/prefix-log.sh


FROM base AS central
RUN wget https://github.com/redis/redis/archive/7.2.4.tar.gz
RUN tar xf 7.2.4.tar.gz
WORKDIR /redis-7.2.4
RUN make -j$(nproc)
RUN pip install redis==5.0.1
RUN pip install tensorboard==2.15.1
COPY central/app /code/app
COPY central/stunnel.conf /tmp/stunnel.conf
COPY central/supervisord.conf /etc/supervisord.conf
WORKDIR /code/app
RUN echo "export UPSD=$(openssl rand -hex 64)" >> /tmp/init.sh
RUN echo "export DPSD=$(openssl rand -hex 64)" >> /tmp/init.sh
RUN echo "export UUSR=$(openssl rand -hex 24)" >> /tmp/init.sh
RUN echo 'sed -e "s/USER/$(echo $UUSR)/g" -e "s/DPASSWD/$(echo $DPSD)/g" -e "s/PASSWD/$(echo $UPSD) /g" /code/app/redisConfig.conf > /code/app/newRedisConf.conf' >> /tmp/init.sh
RUN echo "/redis-7.2.4/src/redis-server /code/app/newRedisConf.conf &" >> /tmp/init.sh
RUN chmod +x /tmp/init.sh
ENTRYPOINT ["/usr/bin/supervisord"]


FROM base AS hie
RUN pip install matplotlib==3.8.2
COPY hie/app /code/app
COPY hie/stunnel.conf /tmp/stunnel.conf
COPY hie/supervisord.conf /etc/supervisord.conf
WORKDIR /code/app
# RUN chmod +x /tmp/init.sh
ENTRYPOINT ["/usr/bin/supervisord"]


