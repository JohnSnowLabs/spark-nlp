# Caps how many model posts one Jekyll wave regenerates.
# Deferred posts are not recorded in .jekyll-metadata, so the next wave retries them.
module BatchLimiter
  module_function

  def limit
    value = Integer(ENV.fetch("JEKYLL_POST_BATCH", "0"))
    value.positive? ? value : nil
  rescue ArgumentError
    nil
  end

  def enabled?
    !limit.nil?
  end

  def allowed
    @allowed ||= 0
  end

  def deferred?
    @deferred == true
  end

  def complete?
    !deferred?
  end

  def allow?(path)
    return true unless enabled?
    return false if deferred?

    if allowed >= limit
      @deferred = true
      warn "Jekyll wave cap reached at #{allowed} posts; remaining posts deferred"
      write_status
      return finish_wave
    end

    @allowed = allowed + 1
    yield path if block_given?
    true
  end

  def finish_wave
    flush_checkpoint
    exit 0 unless ENV["JEKYLL_WAVE_NO_EXIT"] == "1"
    false
  end

  def flush_checkpoint
    flush_metadata
    flush_backups
  end

  def flush_metadata
    return unless defined?(Jekyll) && Jekyll.respond_to?(:sites)

    site = Jekyll.sites&.first
    site&.regenerator&.write_metadata
  rescue StandardError => error
    warn "Unable to flush Jekyll metadata before wave exit: #{error}"
    exit 1
  end

  def flush_backups
    return unless defined?(Jekyll) && Jekyll.respond_to?(:sites)

    site = Jekyll.sites&.first
    return unless site

    source = site.config["source"]
    write_json(File.join(source, "backup-models.json"), models_json)
    write_json(File.join(source, "backup-benchmarking.json"), models_benchmarking_json)
    write_json(File.join(source, "backup-references.json"), models_references_json)
  rescue StandardError => error
    warn "Unable to flush Jekyll backups before wave exit: #{error}"
    exit 1
  end

  def write_json(path, value)
    File.write(path, (value || {}).to_json)
  end

  def write_status
    return unless ENV["GITHUB_OUTPUT"]

    File.open(ENV["GITHUB_OUTPUT"], "a") do |file|
      file.puts "jekyll_wave_complete=#{complete?}"
    end
  end
end
