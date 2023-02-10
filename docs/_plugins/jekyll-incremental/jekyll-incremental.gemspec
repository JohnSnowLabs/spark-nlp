# frozen_string_literal: true

$LOAD_PATH.unshift(File.expand_path("lib", __dir__))
require "jekyll/incremental/version"

Gem::Specification.new do |spec|
  spec.require_paths = ["lib"]
  spec.name = "jekyll-incremental"
  spec.version = Jekyll::Incremental::VERSION
  spec.authors = ["Kshitiz Shakya"]
  spec.email = ["kshitiz@johnsnowlabs.com"]
  spec.files = %w(Rakefile Gemfile README.md)  + Dir["lib/**/*"]
  spec.summary = "A rewritten incremental regeneration based on MD5 digest"
  spec.required_ruby_version = ">= 2.6.0"
  spec.add_runtime_dependency("jekyll", "~> 3.9")
end
