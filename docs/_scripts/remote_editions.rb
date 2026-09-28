require 'json'
require 'net/http'
require 'set'
require 'timeout'
require 'uri'

module RemoteEditions
  class UnavailableError < StandardError; end

  def self.fetch(url, timeout_seconds: 45)
    uri = URI(url)
    response = Timeout.timeout(timeout_seconds) do
      Net::HTTP.start(uri.host, uri.port,
                      use_ssl: uri.scheme == 'https',
                      open_timeout: [timeout_seconds, 10].min,
                      read_timeout: [timeout_seconds, 20].min) do |http|
        http.get(uri.request_uri)
      end
    end
    unless response.is_a?(Net::HTTPSuccess)
      raise UnavailableError, "Remote edition lookup returned HTTP #{response.code}"
    end

    data = JSON.parse(response.body)
    editions = data.is_a?(Hash) ? data.dig('meta', 'aggregations', 'editions') : nil
    raise UnavailableError, 'Remote edition lookup returned an invalid response' unless editions.is_a?(Array)
    raise UnavailableError, 'Remote edition lookup returned no editions' if editions.empty?
    editions.to_set
  rescue Timeout::Error
    raise UnavailableError, "Remote edition lookup timed out after #{timeout_seconds} seconds"
  rescue JSON::ParserError, TypeError
    raise UnavailableError, 'Remote edition lookup returned an invalid response'
  end
end
