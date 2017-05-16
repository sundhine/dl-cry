module NumCry
  extend self

  def zeros(m : Matrix)
    Matrix.new(m.@x, m.@y) do |_, _|
      0.0
    end
  end

  class Matrix
    @b : Pointer(Float64)
    @[AlwaysInline]
    protected def lookup(x, y) : Int32
      (x * @y) + y
    end

    def initialize(input_matrix : Array(Array(Float64)))
      x = input_matrix.size

      a = input_matrix.first.size
      input_matrix.skip(1).each { |arr|
        if arr.size != a
          raise "BAD ARRAY: " + input_matrix.to_s
        end
      }
      y = a

      initialize(x, y) do |x, y|
        input_matrix[x][y]
      end
    end

    def initialize(@x : Int32, @y : Int32)
      @b = Array(Float64).new(@x * @y).to_unsafe

      x = 0
      while x < @x
        y = 0
        while y < @y
          @b[lookup(x, y)] = yield x, y
          y += 1
        end
        x += 1
      end
    end

    def -
      Matrix.new(@x, @y) do |x, y|
        @b[lookup(x, y)] * -1
      end
    end

    def +(other : Number)
      Matrix.new(@x, @y) do |x, y|
        @b[lookup(x, y)] + other
      end
    end

    def +(other : Matrix)
      if @x != other.@x || @y != other.@y
        raise @b.to_s + " " + other.@b.to_s
      end

      Matrix.new(@x, @y) do |x, y|
        arr_lookup = lookup(x, y)
        @b[arr_lookup] + other.@b[arr_lookup]
      end
    end

    def -(other : Number)
      Matrix.new(@x, @y) do |x, y|
        @b[lookup(x, y)] - other
      end
    end

    def -(other : Matrix)
      if @x != other.@x || @y != other.@y
        raise @b.to_s + " " + other.@b.to_s
      end

      Matrix.new(@x, @y) do |x, y|
        arr_lookup = lookup(x, y)
        @b[arr_lookup] - other.@b[arr_lookup]
      end
    end

    def /(other)
      Matrix.new(@x, @y) do |x, y|
        @b[lookup(x, y)] / other
      end
    end

    def *(other : Matrix)
      if @x != other.@x || @y != other.@y
        raise @b.to_s + " " + other.@b.to_s
      end

      Matrix.new(@x, @y) do |x, y|
        arr_lookup = lookup(x, y)
        @b[arr_lookup] * other.@b[arr_lookup]
      end
    end

    def *(other : Number)
      Matrix.new(@x, @y) do |x, y|
        @b[lookup(x, y)] * other
      end
    end

    def exp
      Matrix.new(@x, @y) do |x, y|
        Math.exp(@b[lookup(x, y)])
      end
    end

    def map
      Matrix.new(@x, @y) do |x, y|
        yield @b[lookup(x, y)]
      end
    end

    def dot(m : Matrix)
      if @x != m.@y
        raise "Cannot times matrices " + self.to_s + " " + m.to_s
      end
      s = @x
      Matrix.new(m.@x, @y) do |x, y|
        n = 0
        sum = 0
        while n < s
          sum += @b[lookup(n, y)] * m.@b[m.lookup(x, n)]
          n += 1
        end
        sum.to_f
      end
    end

    def transpose
      Matrix.new(@y, @x) do |x, y|
        @b[lookup(y, x)]
      end
    end

    def argmax_y
      if (@x != 1)
        raise "Must be a vector"
      end
      _, index = (0...@y).reduce({Float64::MIN, -1}) do |acc, i|
        counter = acc[0]
        arr_i = lookup(0, i)
        counter > @b[arr_i] ? acc : {@b[lookup(0, i)], i}
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
        if @b[i] != m.@b[i]
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
          io << @b[lookup(i, j)]
          io << ", "
        }
        io << "]"
      }
      io << "]"
    end
  end
end
