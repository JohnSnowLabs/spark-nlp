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
    .uniq \
    .select { |v| v.start_with?("docs/_posts") } \
    .map { |v| File.basename(v) }

  Jekyll::Hooks.register :posts, :post_render do |post|
    next unless posts.include? post.basename
    posts.delete(post.basename)

    m = !post.data['edition'].nil? && /(.*)\.(\d+)\z/.match(post.data['edition'])
    edition_short = m.is_a?(MatchData) ? m[1] : ''

    body = Nokogiri::HTML(post.content)
    body.search('h2, pre, table, .btn-box, .tabs-box, .highlight').remove
    body = body.text.gsub(/\s+/, ' ').strip

    puts "Indexing #{post.url}..."
    client.index index: 'models', id: post.url, body: {
      name: post.data['name'],
      title: post.data['title'],
      tags_glued: post.data['tags'].join(' '),
      task: post.data['task'],
      language: post.data['language'],
      edition: post.data['edition'],
      edition_short: edition_short,
      date: post.data['date'].strftime('%F'),
      supported: !!post.data['supported'],
      body: body
    }
  end

  Jekyll::Hooks.register :site, :post_render do |site|
    # remove renamed or deleted posts from the index
    posts.map do |filename|
      filename.gsub! /\A(\d{4})-(\d{2})-(\d{2})-(\w+)\.md\z/, '/\1/\2/\3/\4.html'
    end.compact.each do |id|
      puts "Removing #{id}..."
      client.delete index: 'models', id: id, ignore: [404]
    end
  end
end
