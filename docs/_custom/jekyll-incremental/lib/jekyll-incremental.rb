module Jekyll
  # --
  # @note we replace theirs in-place.
  # @example bundle exec jekyll b --incremental
  # A replacement of Jekyll's Regenerator.  That does a
  #   few things a bit differently, most things.
  # --
  module Incremental
    def digest(path)
      return Digest::MD5.file(path).hexdigest
    end

    def add(path)
      return true unless File.exist?(path)

      metadata[path] = {
        "digest" => digest(path),
        "deps"  => [],
      }
      cache[path] = true
    end

    private
    def existing_file_modified?(path)
      # If one of this file dependencies have been modified,
      # set the regeneration bit for both the dependency and the file to true
      metadata[path]["deps"].each do |dependency|
        if modified?(dependency)
          return cache[dependency] = cache[path] = true
        end
      end

      if File.exist?(path) && metadata[path]["digest"].eql?(digest(path))
        # If this file has not been modified, set the regeneration bit to false
        cache[path] = false
      else
        # If it has been modified, set it to true
        add(path)
      end
    end
  end
end

# --
module Jekyll
  # --
  # Patches Jekyll's own regenerator, and replaces it with
  #   our regenerator, which should in theory be more efficient
  #   than Jekyll's since it does less work.
  # --
  class Regenerator
    prepend Jekyll::Incremental
  end
end
