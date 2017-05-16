module NumCry
  extend self

  def zeros(m : Matrix)
    Matrix.new(m.@x, m.@y) do |_, _|
      0.0
    end
  end

  class Matrix
    @[AlwaysInline]
    protected def lookup(x, y) : Int32
      (x * @y) + y
    end

    def initialize(input_matrix : Array(Array(Float64)))
      x = input_matrix.size

      a = input_matrix.first.size
      input_matrix.skip(1).each { |arr|
        if arr.size != a
          raise "BAD ARRAY: " + @matrix.to_s
        end
      }
      y = a

      initialize(x, y) do |x, y|
        input_matrix[x][y]
      end
    end

    def initialize(@x : Int32, @y : Int32)
      @matrix = Array(Float64).new(@x * @y, 0.0)
      b = @matrix.to_unsafe
      (0...@x).each { |xi|
        (0...@y).each { |yi|
          b[lookup(xi, yi)] = yield xi, yi
        }
      }
    end

    def -
      Matrix.new(@x, @y) do |x, y|
        @matrix[lookup(x, y)] * -1
      end
    end

    def +(other : Number)
      Matrix.new(@x, @y) do |x, y|
        @matrix[lookup(x, y)] + other
      end
    end

    def +(other : Matrix)
      if @x != other.@x || @y != other.@y
        raise @matrix.to_s + " " + other.@matrix.to_s
      end

      Matrix.new(@x, @y) do |x, y|
        arr_lookup = lookup(x, y)
        @matrix[arr_lookup] + other.@matrix[arr_lookup]
      end
    end

    def -(other : Number)
      Matrix.new(@x, @y) do |x, y|
        @matrix[lookup(x, y)] - other
      end
    end

    def -(other : Matrix)
      if @x != other.@x || @y != other.@y
        raise @matrix.to_s + " " + other.@matrix.to_s
      end

      Matrix.new(@x, @y) do |x, y|
        arr_lookup = lookup(x, y)
        @matrix[arr_lookup] - other.@matrix[arr_lookup]
      end
    end

    def /(other)
      Matrix.new(@x, @y) do |x, y|
        @matrix[lookup(x, y)] / other
      end
    end

    def *(other : Matrix)
      if @x != other.@x || @y != other.@y
        raise @matrix.to_s + " " + other.@matrix.to_s
      end

      Matrix.new(@x, @y) do |x, y|
        arr_lookup = lookup(x, y)
        @matrix[arr_lookup] * other.@matrix[arr_lookup]
      end
    end

    def *(other : Number)
      Matrix.new(@x, @y) do |x, y|
        @matrix[lookup(x, y)] * other
      end
    end

    def exp
      Matrix.new(@x, @y) do |x, y|
        Math.exp(@matrix[lookup(x, y)])
      end
    end

    def map
      Matrix.new(@x, @y) do |x, y|
        yield @matrix[lookup(x, y)]
      end
    end

    def dot(m : Matrix)
      if @x != m.@y
        raise "Cannot times matrices " + self.to_s + " " + m.to_s
      end
      s = @x
      Matrix.new(m.@x, @y) do |x, y|
        (0...s).map { |n| @matrix[lookup(n, y)] * m.@matrix[m.lookup(x, n)] }
               .reduce { |a, b| a + b }
      end
    end

    def transpose
      Matrix.new(@y, @x) do |x, y|
        @matrix[lookup(y, x)]
      end
    end

    def argmax_y
      if (@x != 1)
        raise "Must be a vector"
      end
      _, index = (0...@y).reduce({Float64::MIN, -1}) do |acc, i|
        counter = acc[0]
        arr_i = lookup(0, i)
        counter > @matrix[arr_i] ? acc : {@matrix[lookup(0, i)], i}
      end

      Matrix.new(1, @y) do |_, y|
        y == index ? 1.0 : 0.0
      end
    end

    def ==(m : Matrix)
      if @x != m.@x
        return false
      end

      if @y != m.@y
        return false
      end

      (0...@x * @y).each do |i|
        if @matrix[i] != m.@matrix[i]
          return false
        end
      end

      return true
    end

    def to_s(io)
      io << "Matrix "
      io << @x.to_s << " x " << @y.to_s
      io << " ["
      (0...@x).each { |i|
        io << "["
        (0...@y).each { |j|
          io << @matrix[lookup(i, j)]
          io << ", "
        }
        io << "]"
      }
      io << "]"
    end
  end
end
