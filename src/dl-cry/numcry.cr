module NumCry
  extend self

  def zeros(m : Matrix)
    Matrix.new(m.@x, m.@y) do |_, _|
      0.0
    end
  end

  class Matrix
    # TODO defensive copy
    def initialize(@matrix : Array(Array(Float64)))
      @x = @matrix.size

      a = @matrix.first.size
      @matrix.skip(1).each { |arr|
        if arr.size != a
          raise "BAD ARRAY: " + @matrix.to_s
        end
      }
      @y = a
    end

    def initialize(@x : Int32, @y : Int32)
      @matrix = Array(Array(Float64)).new(x)
      (0...x).each { |xi|
        arr = Array(Float64).new(y)
        (0...y).each { |yi|
          arr << yield xi, yi
        }
        @matrix << arr
      }
    end

    def -
      Matrix.new(@x, @y) do |x, y|
        @matrix[x][y] * -1
      end
    end

    def +(other : Number)
      Matrix.new(@x, @y) do |x, y|
        @matrix[x][y] + other
      end
    end

    def +(other : Matrix)
      if @x != other.@x || @y != other.@y
        raise @matrix.to_s + " " + other.@matrix.to_s
      end

      Matrix.new(@x, @y) do |x, y|
        @matrix[x][y] + other.@matrix[x][y]
      end
    end

    def -(other : Number)
      Matrix.new(@x, @y) do |x, y|
        @matrix[x][y] - other
      end
    end

    def -(other : Matrix)
      if @x != other.@x || @y != other.@y
        raise @matrix.to_s + " " + other.@matrix.to_s
      end

      Matrix.new(@x, @y) do |x, y|
        @matrix[x][y] - other.@matrix[x][y]
      end
    end

    def /(other)
      Matrix.new(@x, @y) do |x, y|
        @matrix[x][y] / other
      end
    end

    def *(other : Matrix)
      if @x != other.@x || @y != other.@y
        raise @matrix.to_s + " " + other.@matrix.to_s
      end

      Matrix.new(@x, @y) do |x, y|
        @matrix[x][y] * other.@matrix[x][y]
      end
    end

    def *(other : Number)
      Matrix.new(@x, @y) do |x, y|
        @matrix[x][y] * other
      end
    end

    def exp
      Matrix.new(@x, @y) do |x, y|
        Math.exp(@matrix[x][y])
      end
    end

    def map
      Matrix.new(@x, @y) do |x, y|
        yield @matrix[x][y]
      end
    end

    def dot(m : Matrix)
      if @x != m.@y
        raise "Cannot times matrices " + self.to_s + " " + m.to_s
      end
      s = @x
      Matrix.new(m.@x, @y) do |x, y|
        (0...s).map { |n| @matrix[n][y] * m.@matrix[x][n] }
               .reduce { |a, b| a + b }
      end
    end

    def transpose
      Matrix.new(@y, @x) do |x, y|
        @matrix[y][x]
      end
    end

    def argmax_y
      if (@x != 1)
        raise "Must be a vector"
      end
      _, index = (0...@y).reduce({Float64::MIN, -1}) do |acc, i|
        counter = acc[0]
        counter > @matrix[0][i] ? acc : {@matrix[0][i], i}
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

      (0...@x).each do |x|
        (0...@y).each do |y|
          if @matrix[x][y] != m.@matrix[x][y]
            return false
          end
        end
      end

      return true
    end

    def to_s(io)
      io << "Matrix "
      io << @x.to_s << " x " << @y.to_s
      io << " ["
      @matrix.each_with_index { |arr, _|
        io << "["
        arr.each_with_index { |n, _|
          io << n
          io << ", "
        }
        io << "]"
      }
      io << "]"
    end
  end
end
