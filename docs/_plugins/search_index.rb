require 'elasticsearch'
require 'nokogiri'

unless ENV['ELASTICSEARCH_URL'].to_s.empty?
  puts "Connecting to Elasticsearch..."
  client = Elasticsearch::Client.new url: ENV['ELASTICSEARCH_URL'], transport_options: {
    headers: {
      'Authorization': "Bearer #{ENV['ELASTICSEARCH_ACCESS_TOKEN']}"
    }
  }

  posts = File.open("./changes.txt") \
    .each_line \
    .to_a \
    .map(&:chomp) \
    .select { |v| v.start_with?("docs/_posts") } \
    .map { |v| File.basename(v) }
  Jekyll::Hooks.register :posts, :post_render do |post|
    next unless posts.include? post.basename
    m = !post.data['edition'].nil? && /(.*)\.(\d+)\z/.match(post.data['edition'])
    edition_short = m.is_a?(MatchData) ? m[1] : ''
    body = Nokogiri::HTML(post.content).xpath('//p[not(contains(@class, "btn-box"))]').map do |p|
      p.text.gsub(/\s+/, ' ')
    end.join(' ')

    puts "Indexing #{post.url}..."
    client.index index: 'models', id: post.url, body: {
      title: post.data['title'],
      task: post.data['task'],
      language: post.data['language'],
      edition: post.data['edition'],
      edition_short: edition_short,
      date: post.data['date'].strftime('%F'),
      supported: !!post.data['supported'],
      body: body
    }
  end
end
