require 'set'
require 'uri'
require 'net/http'
require 'json'
require 'elasticsearch'
require 'nokogiri'

class Version < Array
  def initialize name
    m = /(\d+\.\d+)\z/.match(name)
    super(m ? m[1].split('.').map(&:to_i) : [])
  end

  def < x
    (self <=> x) < 0
  end

  def <= x
    (self <=> x) <= 0
  end
end

def compatible_editions(editions, model_editions, edition)
  return edition if edition.to_s.empty?

  outdated_editions = ['Spark NLP 2.1', 'Spark NLP for Healthcare 2.0']
  return edition if outdated_editions.include? edition

  for_healthcare = edition.include?('Healthcare')
  editions = editions.select do |v|
    for_healthcare == v.include?('Healthcare') && !outdated_editions.include?(v)
  end
  model_editions = model_editions.select { |v| for_healthcare == v.include?('Healthcare') }

  curr_index = model_editions.index(edition)
  return edition if curr_index.nil?

  min = edition
  max = model_editions[curr_index + 1].nil? ? nil : model_editions[curr_index + 1]
  editions.select do |v|
    if min && max
      (Version.new(min) <= Version.new(v)) && (Version.new(v) < Version.new(max))
    elsif max
      Version.new(v) < Version.new(max)
    else
      Version.new(min) <= Version.new(v)
    end
  end
end

def editions_changed?(local_editions)
  uri = URI('https://search.modelshub.johnsnowlabs.com/')
  res = Net::HTTP.get_response(uri)
  if res.is_a?(Net::HTTPSuccess)
    data = JSON.parse(res.body)
    remote_editions = data['meta']['aggregations']['editions']
    return !local_editions.to_set.subset?(remote_editions.to_set)
  end
  true
end

class Extractor
  def initialize(content)
    @content = content
  end

  def download_link
    m = /\[Download\]\((.*?)\)/.match(@content)
    m ? m[1] : nil
  end

  def predicted_entities
    m = /## Predicted Entities(.*?)(##|{:\.btn-box\})/m.match(@content)
    if m
      buf = m[1].strip
      strategies = ['newline_separated', 'comma_separated']
      while strategies.length > 0 do
        case strategies.pop
        when 'comma_separated'
          items = self.comma_separated_predicted_entities(buf)
        when 'newline_separated'
          items = self.newline_separated_predicted_entities(buf)
        end
        return items if items
      end
    end
    nil
  end

  private

  def comma_separated_predicted_entities(buf)
    has_comma = buf.include?(',')
    has_apostrophe = buf.include?('`')
    buf = buf.gsub("\n", ' ')
    return nil unless has_comma || (has_apostrophe && buf.match?(/\A\S+\z/))
    buf.scan(/`[^`]+`/).map { |v| v.gsub(/["`]/, '') }.select { |v| !v.empty? && !v.include?(',') }
  end

  def newline_separated_predicted_entities(buf)
    return nil unless buf.include? "\n"
    return nil if buf.strip.start_with? '-'
    buf.split("\n").collect { |v| v.gsub(/^-\s?/, '').strip }.select { |v| !v.empty? }
  end
end

class BulkIndexer
  def initialize(client)
    @client = client
    @buffer = []
  end

  def index(id, data)
    @buffer << { index: { _id: id, data: data } }
    self.execute if @buffer.length >= 100
  end

  def execute
    return nil unless @client
    return nil if @buffer.empty?
    puts "Indexing #{@buffer.length} models..."
    @client.bulk(index: 'models', body: @buffer)
    @buffer.clear
  end
end

editions = Set.new
uniq_to_models_mapping = {}
uniq_for_indexing = Set.new
models_json = {}

changed_filenames = []
if File.exist?("./changes.txt")
  changed_filenames = File.open("./changes.txt") \
    .each_line \
    .to_a \
    .map(&:chomp) \
    .uniq \
    .select { |v| v.start_with?("docs/_posts") } \
    .map { |v| File.basename(v) }
end

Jekyll::Hooks.register :posts, :pre_render do |post|
  extractor = Extractor.new(post.content)

  models_json[post.url] = {
    title: post.data['title'],
    date: post.data['date'].strftime('%B %d, %Y'),
    name: post.data['name'],
    class: post.data['class'] || "",
    language: post.data['language'],
    task: post.data['task'],
    edition: post.data['edition'],
    categories: post.data['categories'],
    url: post.url,
    tags: post.data['tags'],
    download_link: extractor.download_link,
    predicted_entities: extractor.predicted_entities || [],
  }
end

Jekyll::Hooks.register :posts, :post_render do |post|
  m = !post.data['edition'].nil? && /(.*)\.(\d+)\z/.match(post.data['edition'])
  edition_short = m.is_a?(MatchData) ? m[1] : ''

  body = Nokogiri::HTML(post.content)
  body.search('h2, pre, table, .btn-box, .tabs-box, .highlight').remove
  body = body.text.gsub(/\s+/, ' ').strip

  model = {
    id: post.url,
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

  uniq = "#{post.data['name']}_#{post.data['language']}"
  uniq_to_models_mapping[uniq] = [] unless uniq_to_models_mapping.has_key? uniq
  uniq_to_models_mapping[uniq] << model
  editions.add(edition_short) unless edition_short.empty?
  uniq_for_indexing << uniq if changed_filenames.include?(post.basename)
  changed_filenames.delete(post.basename)
end

client = nil
unless ENV['ELASTICSEARCH_URL'].to_s.empty?
  puts "Connecting to Elasticsearch..."
  client = Elasticsearch::Client.new(
    url: ENV['ELASTICSEARCH_URL'],
    transport_options: {
      headers: {
        'Authorization': "Bearer #{ENV['ELASTICSEARCH_ACCESS_TOKEN']}"
      },
      request: {
        open_timeout: 60,
        timeout: 60,
      },
    },
  )
end

Jekyll::Hooks.register :site, :post_render do |site|
  force_reindex = editions_changed?(editions)
  bulk_indexer = BulkIndexer.new(client)

  uniq_to_models_mapping.each do |uniq, items|
    items.sort_by! { |v| v[:edition_short] }
    model_editions = items.map { |v| v[:edition_short] }.uniq
    items.each do |model|
      next_edition_short = compatible_editions(
        editions,
        model_editions,
        model[:edition_short]
      )
      model[:edition_short] = next_edition_short
      models_json[model[:id]][:compatible_editions] = next_edition_short.empty? ? [] : Array(next_edition_short)

      if client
        if force_reindex || uniq_for_indexing.include?(uniq)
          id = model.delete(:id)
          bulk_indexer.index(id, model)
        end
      end
    end
  end
  bulk_indexer.execute

  # remove renamed or deleted posts from the index
  if client
    changed_filenames.map do |filename|
      filename.gsub! /\A(\d{4})-(\d{2})-(\d{2})-(\w+)\.md\z/, '/\1/\2/\3/\4.html'
    end.compact.each do |id|
      puts "Removing #{id}..."
      client.delete index: 'models', id: id, ignore: [404]
    end
  end
end

Jekyll::Hooks.register :site, :post_write do |site|
  filename = File.join(site.config['destination'], 'models.json')
  File.write(filename, models_json.values.to_json)
end
