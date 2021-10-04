FROM ruby:2.7.3-slim

RUN apt-get update && apt-get install -y make gcc g++

RUN mkdir -p /docs
COPY . /docs
WORKDIR /docs

RUN bundle update
RUN bundle install

CMD ["bash" , "-c" , "bundle update; bundle install; bundle exec jekyll serve --incremental --trace"]
