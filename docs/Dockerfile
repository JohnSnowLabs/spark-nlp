FROM ruby:3.1
WORKDIR /app

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
COPY Gemfile ./Gemfile
COPY Gemfile.lock ./Gemfile.lock
COPY _frontend ./_frontend

RUN curl -sL https://deb.nodesource.com/setup_14.x | bash -\
  && apt-get update -qq && apt-get install -qq --no-install-recommends \
    nodejs \
  && apt-get upgrade -qq \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*\
  && npm install -g yarn@1

RUN bundle install
WORKDIR /app/_frontend
RUN yarn && yarn run build
RUN npm install -g gh-pages@3.0.0
WORKDIR /docs
