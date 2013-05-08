usage: fractal [-h] --position pos pos --dimensions dim dim --zoom ZOOM
               [--scale SCALE] [--processes PROCESSES] [-m]
               [--iterations ITERS] [--calculate] [--blocksize BSIZE]
               name

A program to generate a fractal image. 
In the case of 2-argument flags (such as --position), the arguments should be space-seperated. As in: --position 10 10

positional arguments:
  name                  The name of the output file. This can have any image
                        extension supported by PIL.

optional arguments:
  -h, --help            show this help message and exit
  --position pos pos, -p pos pos
                        The offset of the rendered fractal. If set to 0 0, the
                        fractal will be centered in the output image.
  --dimensions dim dim, -d dim dim
                        The rendered image dimensions, in pixels. This may be
                        modified by the use of the scale argument.
  --zoom ZOOM, -z ZOOM  The zoom of the fractal. This may be modified by the
                        scale argument.
  --scale SCALE, -s SCALE
                        The scale multiplier. 1=default, use provided position
                        and zoom. 2=2x zoom and position. If used with the -m
                        argument, the output dimensions will also be scaled
                        up. This can be used to make a thumbnail image before
                        running the full render, to see if your coordinates
                        are correct.
  --processes PROCESSES, -procs PROCESSES
                        The number of processes to use for multi-process
                        rendering. This is usually the number of CPU cores you
                        have.
  -m                    If this flag is set, then the scale argument will also
                        modify the dimensions of the image.
  --iterations ITERS, -i ITERS
                        The maximum number of iterations the generator will go
                        through, per pixel. The higher this is, the more
                        accurate the fractal will be, and the slower the
                        generation will be. It will be especially slow in
                        places where the pixel is inside the shape, in which
                        case the loop would go on for ever without a limit.
  --calculate           If this flag is set, no fractal will be rendered.
                        Instead, scale will be applied, and the resulting
                        values will be printed out, so that you can use them
                        without using scale. (Work on this description...)
  --blocksize BSIZE     Set the block generation size. Default 50.

Example usage:
        ./fractal.py -p 0 0 -z 50 -d 200 200 -- This will render a 200x200 image, with the fractal centered, and zoomed to 50.

        ./fractal.py -p 100 0  -d 200 200 -z 50 -s 2 -- This will render a 200x200 image, zoomed to 50, and then scaled up 2x. 
        Without the -m argument this effectively means it will zoom in on the center of the image 2x.

        If you have a case such as this: 
        ./fractal_2.py -p 0 -29.65 -d 400 1500 -z 20 -s 230 fractal.png
        And you want to render it at 2x resolution, you can't simply add '-s 2 -m' because you already have scale defined and by adding -m, 
        you would be rendering it at 230x resolution.
        So this is where the --calculate option is useful. This will calculate the modied position and zoom arguments, 
        and print out a command that will output the same as your current setup, without the scale argument like so:
        ./fractal_2.py -p 0 -6819 -d 400 1500 -z 4600.0
        You can then add the scale and -m arguments as you like to scale your render by whatever you want.
