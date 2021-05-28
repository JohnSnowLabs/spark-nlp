require 'json'

class Extractor
  def initialize(content)
    @content = content
  end

  def download_link
    m = /\[Download\]\((.*?)\)/.match(@content)
    m ? m[1] : nil
  end

  def predicted_entities
    m = /## Predicted Entities\n(.*?)\{:\.btn-box\}/m.match(@content)
    if m
      buf = m[1].strip
      strategies = ['comma_separated', 'newline_separated']
      while strategies.length > 0 do
        case strategies.pop
        when 'comma_separated'
          items = self.comma_separated_predicted_entities(buf)
        when 'newline_separated'
          items = self.newline_separated_predicted_entities(buf)
        end
        return items if items.kind_of?(Array) and items.all? { |v| self.valid_predicted_entity? v }
      end
    end
    nil
  end

  private

  def comma_separated_predicted_entities(buf)
    return nil unless buf.include? ','
    buf.split(',').collect { |v| v.strip.gsub(/[`\.]/, '') }
  end

  def newline_separated_predicted_entities(buf)
    return nil unless buf.include? "\n"
    buf.split("\n").collect { |v| v.strip.gsub(/^-\s?/, '').gsub(/[`\.]/, '') }
  end

  def valid_predicted_entity?(v)
    /^[A-Za-z0-9_-]+$/.match? v
  end
end

models = []

Jekyll::Hooks.register :posts, :pre_render do |post|
  extractor = Extractor.new(post.content)

  download_link = extractor.download_link
  predicted_entities = extractor.predicted_entities

  models << {
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
    download_link: download_link,
    predicted_entities: predicted_entities || [],
  }
end

Jekyll::Hooks.register :site, :post_write do |site|
  filename = File.join(site.config['destination'], 'models.json')
  File.write(filename, models.to_json)
end
