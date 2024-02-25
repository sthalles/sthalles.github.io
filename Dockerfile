FROM ubuntu:focal

RUN apt-get update
RUN apt-get install -y ruby-full build-essential zlib1g-dev

SHELL ["/bin/bash", "-c"] 
RUN echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
RUN echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
RUN echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
RUN source ~/.bashrc

RUN gem install jekyll -v 3.1.2
RUN gem install bundler -v 2.4.22

COPY ./ /workspace/sthalles.github.io/
WORKDIR /workspace/sthalles.github.io/
RUN bundle install

ARG UID=10264
ARG GID=1000
ARG GNAME=thalles
ARG USERNAME=thalles
RUN if [ $GNAME != users ]; then groupadd --gid $GID $GNAME; fi && \
    useradd --create-home --shell /bin/bash --uid $UID --gid $GID $USERNAME && \
    echo "$USERNAME:$USERNAME" | chpasswd && \
    adduser $USERNAME sudo

CMD ["bundle", "exec", "jekyll", "serve", "--livereload", "--host", "0.0.0.0"]
