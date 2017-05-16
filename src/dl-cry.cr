require "./dl-cry/*"
include NumCry

module Dl::Cry
  RAN = Random.new

  def get_normal_random_matrix(x, y)
    Matrix.new(x, y) do
      Zig.rand_norm(RAN)
    end
  end

  def sigmoid(m)
    ((-m).exp + 1).map do |f|
      1 / f
    end
  end

  def sigmoid_prime(z)
    sg = sigmoid(z)
    sg * (sg.map { |f| 1 - f })
  end

  alias TrainingData = Array({Matrix, Matrix})

  class Network
    @num_layers : Int32
    @sizes : Array(Int32)
    @biases : Array(Matrix)
    @weights : Array(Matrix)

    def initialize(sizes : Array(Int32))
      @num_layers = sizes.size
      @sizes = sizes
      @biases = sizes.skip(1).map { |n| get_normal_random_matrix(1, n) }
      @weights = sizes.first(sizes.size - 1).zip(sizes.skip(1)).map { |t| get_normal_random_matrix(t[0], t[1]) }
    end

    def feedforward(input : Matrix)
      @biases.zip(@weights).reduce(input) { |i, t|
        sigmoid(t[1].dot(i) + t[0])
      }
    end

    def sgd(training_data, epochs, mini_batch_size, eta, test_data)
      n_test = test_data.size
      (1..epochs).each { |j|
        training_data.shuffle!
        training_data.each_slice(mini_batch_size).each { |set|
          update_mini_batch(set, eta)
        }
        puts "Epoch #{j}: #{evaluate(test_data)} / #{n_test}"
      }
    end

    def evaluate(test_data)
      test_results = test_data.map { |t|
        x, y = t[0], t[1]
        result = feedforward(x).argmax_y
        result == y ? 1 : 0
      }.reduce do |x, y|
        x + y
      end
    end

    private def update_mini_batch(mini_batch, eta)
      nabla_b = @biases.map { |b| NumCry.zeros(b) }
      nabla_w = @weights.map { |w| NumCry.zeros(w) }

      mini_batch.each { |t|
        x, y = t[0], t[1]
        delta_nabla_b, delta_nable_w = backprop(x, y)
        nabla_b = nabla_b.zip(delta_nabla_b).map { |t| t[0] + t[1] }
        nabla_w = nabla_w.zip(delta_nable_w).map { |t| t[0] + t[1] }
      }
      @weights = @weights.zip(nabla_w).map { |t|
        w, nw = t[0], t[1]
        (nw * (eta / mini_batch.size)) + (-w)
      }
      @biases = @biases.zip(nabla_b).map { |t|
        b, nb = t[0], t[1]
        (nb * (eta / mini_batch.size)) + (-b)
      }
    end

    private def backprop(x, y) : {Array(Matrix), Array(Matrix)}
      nabla_b = @biases.map { |b| NumCry.zeros(b) }
      nabla_w = @weights.map { |w| NumCry.zeros(w) }
      # feedforward
      activation = x
      activations = [x] # list to store all the activations, layer by layer
      zs = [] of Matrix # list to store all the z vectors, layer by layer
      @biases.zip(@weights).each { |t|
        b, w = t[0], t[1]
        z = w.dot(activation) + b
        zs << z
        activation = sigmoid(z)
        activations << activation
      }
      # backward pass
      delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
      nabla_b[-1] = delta
      nabla_w[-1] = delta.dot(activations[-2].transpose)
      (2...@num_layers).each { |l|
        z = zs[-l]
        sp = sigmoid_prime(z)
        delta = @weights[-l + 1].transpose.dot(delta) * sp
        nabla_b[-l] = delta
        nabla_w[-l] = delta.dot(activations[-l - 1].transpose)
      }
      {nabla_b, nabla_w}
    end

    private def cost_derivative(output_activations, y)
      output_activations - y
    end

    def to_s(io)
      io << "Network"
      io << "\nNumber of layers: "
      io << @num_layers
      io << "\nSizes: "
      io << @sizes
    end
  end
end

module DataProcessing
  extend self

  def data
    {
      training_data:   read_data("data/tr.csv"),
      validation_data: read_data("data/va.csv"),
      test_data:       read_data("data/te.csv"),
    }
  end

  private def read_data(file)
    File.each_line(file).map { |st|
      arr = st.split(",").map { |s| s.strip }
      {
        format_data(arr.skip(1)),
        vectorize(arr[0].to_i),
      }
    }.to_a
  end

  private def vectorize(n)
    Matrix.new(1, 10) do |_, y|
      y == n - 1 ? 1.0 : 0.0
    end
  end

  private def format_data(ns)
    Matrix.new(1, ns.size) do |_, y|
      ns[y].to_f
    end
  end
end

include Dl::Cry

# m1 = Matrix.new(3, 4) do |x, y|
#   (x + y).to_f
# end

# m2 = Matrix.new(6, 3) do |x, y|
#   (x + y).to_f
# end

# puts m1.dot(m2)

puts "Loading data..."
data = DataProcessing.data
puts "...data loaded"

net = Network.new([784, 30, 10])
net.sgd(data[:training_data], 30, 10, 3.0, data[:test_data])
