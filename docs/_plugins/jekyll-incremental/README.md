# Jekyll Incremental

Jekyll Incremental is a rewrite of the core incremental regeneration feature. It replaces the default incremental logic that is based on the posts modified time. It uses MD5 digest of the post content to compare changes to posts for incremental regeneration.

See these for more reference:

https://github.com/jekyll/jekyll/pull/9166
https://github.com/jayvdb/jekyll-incremental

## Installation

```ruby
group "jekyll-plugins" do
    gem "jekyll-incremental", "0.1.0", path: "_plugins/jekyll-incremental"
end
```
Also add this to the `_config.yml`
```ruby
plugins:
  - jekyll-incremental
```

## Usage

```ruby
bundle exec jekyll build --incremental
```
