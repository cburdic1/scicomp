using WaveSim, Plots

# This is how the animation at https://byuhpc.github.io/sci-comp-course/project/overview was created.

# Initialize the grid
w = WaveOrthotope(0.01, 0, 25, 50)
w.v[2:end-1,2:end-1] .= 0.1

# Create the gif
animation = @gif while energy(w) > prod(size(w).-2)/1000
    surface(w.u, clims=(-8, 8), zlims=(-8, 8), camera=(45, 15), grid=false, colorbar=false, axis=false, ticks=1:0, aspect_ratio=1, size=(2000, 2000))
    step!(w)
end every 250

# Crop the gif since `surface` leaves huge margins
run(`convert $(animation.filename) -coalesce -repage 0x0 -crop 1200x400+400+800 +repage phase-1-animation.gif`)
