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
      return false
    end

    @allowed = allowed + 1
    yield path if block_given?
    true
  end

  def write_status
    return unless ENV["GITHUB_OUTPUT"]

    File.open(ENV["GITHUB_OUTPUT"], "a") do |file|
      file.puts "jekyll_wave_complete=#{complete?}"
    end
  end
end
