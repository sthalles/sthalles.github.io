FROM ubuntu:focal

RUN apt-get update
RUN apt-get install -y build-essential ruby-full

COPY ./ /workspace/blog
WORKDIR /workspace/blog

RUN gem install jekyll bundler
RUN bundle install

CMD ["bundle", "exec", "jekyll", "serve", "--livereload", "--host", "0.0.0.0"]

ARG UID=10264
ARG GID=1000
ARG GNAME=thalles
ARG USERNAME=thalles
RUN if [ $GNAME != users ]; then groupadd --gid $GID $GNAME; fi && \
    useradd --create-home --shell /bin/bash --uid $UID --gid $GID $USERNAME && \
    echo "$USERNAME:$USERNAME" | chpasswd && \
    adduser $USERNAME sudo

