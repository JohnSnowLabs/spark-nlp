require 'set'
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

unless ENV['ELASTICSEARCH_URL'].to_s.empty?
  puts "Connecting to Elasticsearch..."
  client = Elasticsearch::Client.new url: ENV['ELASTICSEARCH_URL'], transport_options: {
    headers: {
      'Authorization': "Bearer #{ENV['ELASTICSEARCH_ACCESS_TOKEN']}"
    }
  }

  changed_filenames = File.open("./changes.txt") \
    .each_line \
    .to_a \
    .map(&:chomp) \
    .uniq \
    .select { |v| v.start_with?("docs/_posts") } \
    .map { |v| File.basename(v) }

  editions = Set.new
  uniq_to_models_mapping = {}
  uniq_for_indexing = Set.new
  force_reindex = false

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
    uniq_for_indexing << uniq if force_reindex || changed_filenames.include?(post.basename)
    changed_filenames.delete(post.basename)
  end

  Jekyll::Hooks.register :site, :post_render do |site|
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

        if uniq_for_indexing.include?(uniq)
          id = model.delete(:id)
          puts "Indexing #{id}..."
          client.index index: 'models', id: id, body: model
        end
      end
    end
  end

  Jekyll::Hooks.register :site, :post_render do |site|
    # remove renamed or deleted posts from the index
    changed_filenames.map do |filename|
      filename.gsub! /\A(\d{4})-(\d{2})-(\d{2})-(\w+)\.md\z/, '/\1/\2/\3/\4.html'
    end.compact.each do |id|
      puts "Removing #{id}..."
      client.delete index: 'models', id: id, ignore: [404]
    end
  end
end
