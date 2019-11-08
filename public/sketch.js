(function() {
  const canvasSizeX = 300;
  const canvasSizeY = 250;

  const selectionSizes = [15, 45, 81, 105, 129];
  // const selectionColors = ['#A74661', '#956C89', '#8490B0', '#71B6D7', '#67CBEF'];
  const selectionColors = ['#A74661', '#67CBEF', '#956C89', '#8490B0', '#71B6D7'];
  const sketches = {
    stored: null,
    temp: null,
    debug: null,
    comp: null,
    comp2: null,
    comp_temp: null,
    selections: {
      p0: [],
      p1: [],
    },
  };
  let drawTimer = null;

  /**
  * Given a before and after at various scales and an image, return
  * suggestions for transformations to the image at those scales.
  */
  function getSuggestions(befores, afters, img, scale, n=4) {
    console.log('fetching suggestions...');
    const data = { befores, afters, img, n };
    console.log('Suggested Layer ' + (scale + 1));

    // console.log(befores);
    // const i = 1;
    // const container = $('#L' + (i + 1) + '_matches .selection_matches.p0');
    // container.empty();
    // const img3 = new Image();
    // img3.src = befores[i];
    // container.append(img3);
    // const container2 = $('#L' + (i + 1) + '_matches .selection_matches.p1');
    // container2.empty();
    // const img2 = new Image();
    // img2.src = afters[i];
    // container2.append(img2);


    $.get('/suggestions', data, function(result) {
      console.log(result);
      sketches.comp.clear();
      sketches.comp.noFill();
      sketches.comp2.clear();
      sketches.comp2.noFill();
      if (drawTimer) {
        clearTimeout(drawTimer);
      }

      Object.keys(result).forEach((key, i) => {
        const { locations, locationImgs, suggestions, suggestionsAligned } = result[key];
          // add suggestions at this scale to canvas
        sketches.comp2.stroke(sketches.comp.color(selectionColors[i]));

        const drawSuggestions = (j) => {
          if (j >= suggestions.length - 2) {
            return;
          }
          drawDataURLToP(sketches.comp, suggestions[j], locations[j]);
          drawTimer = setTimeout(() => {
            const comp = sketches.comp.get();
            sketches.stored.image(comp, 0, 0);
            sketches.comp.clear();
            drawSuggestions(j + 1);
          }, 1000);
        }
        if (i === 1) {
          drawSuggestions(0);
        }

        locations.forEach((location, j) => {
          const { x, y } = location;
          sketches.comp2.rect(x, y, selectionSizes[i], selectionSizes[i]);

          // drawDataURLToP(sketches.comp, suggestions[j], location);
          // setTimeout(() => {
          //   const comp = sketches.comp.get();
          //   sketches.stored.image(comp, 0, 0);
          //   sketches.comp.clear();
          // }, 500);
        });

        // draw suggestions to side bar
        if (locationImgs) {
          console.log('update locations')
          const container = $('#L' + (i + 1) + '_matches .selection_matches.p0');
          container.empty();
          locationImgs.forEach(dataURL => {
            const img = new Image();
            img.src = 'data:image/png;base64,' + dataURL;
            container.append(img);
          });
        }
        if (suggestionsAligned) {
          const container = $('#L' + (i + 1) + '_matches .selection_matches.p1');
          container.empty();
          suggestionsAligned.forEach(dataURL => {
            const img = new Image();
            img.src = 'data:image/png;base64,' + dataURL;
            container.append(img);
          });
        }
        if (suggestions) {
          const containerBase = $('#L' + (i + 1) + '_matches .selection_matches.p2.base');
          containerBase.empty();
          locationImgs.forEach(dataURL => {
            const img = new Image();
            img.src = 'data:image/png;base64,' + dataURL;
            containerBase.append(img);
          });
          const containerOverlay = $('#L' + (i + 1) + '_matches .selection_matches.p2.overlay');
          containerOverlay.empty();
          suggestions.forEach(dataURL => {
            const img = new Image();
            img.src = 'data:image/png;base64,' + dataURL;
            containerOverlay.append(img);
          });
        }
      })
    });
  }

  function drawDataURLToP(p, dataURL, location) {
    const raw = new Image();
    raw.src = 'data:image/jpeg;base64,' + dataURL;
    const { x, y } = location;
    raw.onload = function() {
      const img = sketches.comp_temp.createImage(raw.width, raw.height);
      img.drawingContext.drawImage(raw, 0, 0);

      // draw to graphics and threshold and make non thresh transparent
      // const pg = createGraphics(raw.width, raw.height);
      // pg.image(img, 0, 0);
      sketches.comp_temp.resizeCanvas(raw.width, raw.height);
      sketches.comp_temp.image(img, 0, 0);
      sketches.comp_temp.loadPixels();
      for (let i = 0; i < sketches.comp_temp.pixels.length; i += 4) {
        const gr = sketches.comp_temp.pixels[i];
        const o = (gr - 255) / -255;

        sketches.comp_temp.pixels[i] = 0;
        sketches.comp_temp.pixels[i + 1] = 0;
        sketches.comp_temp.pixels[i + 2] = 0;
        if (o < 0.25) {
          sketches.comp_temp.pixels[i + 3] = 0;
        } else {
          sketches.comp_temp.pixels[i + 3] = ((o * 0.3) + 0.7) * 255;
        }
      }
      sketches.comp_temp.updatePixels();
      const img_f = sketches.comp_temp.get();

      p.image(img_f, x, y);
    }
  }

  function getChangeSelections(p, bounds, lastPos) {
    const [ bStartX, bStartY, bEndX, bEndY ] = bounds.map(bound => Math.floor(bound));
    const minPad = 0;
    const [ lastPosX, lastPosY ] = lastPos.map(v => Math.floor(v));

    // get selections containing end of edit (with a little padding) centered as much as possible
    const selectionBoundsBySize = selectionSizes.map(selectionSize => {
      // padding to make selection size
      let padToDistribute = selectionSize - 1;

      // add min padding
      const selectionBounds = [lastPosX - minPad, lastPosY - minPad, lastPosX + minPad + 1, lastPosY + minPad + 1];
      padToDistribute -= 2 * minPad;
      // distribute padding where room, then equally
      while (padToDistribute > 0) {
        // if more space or other side is maxxed out
        if (selectionBounds[0] > bStartX || selectionBounds[2] >= bEndX) {
          selectionBounds[0] -= 1;
        } else {
          // give to other side, since still room
          selectionBounds[2] += 1;
        }
        if (selectionBounds[2] < bEndX || selectionBounds[0] <= bStartX) {
          selectionBounds[2] += 1;
        } else {
          selectionBounds[0] -= 1;
        }

        if (selectionBounds[1] > bStartY || selectionBounds[3] >= bEndY) {
          selectionBounds[1] -= 1;
        } else {
          // give to other side, since still room
          selectionBounds[3] += 1;
        }
        if (selectionBounds[3] < bEndY || selectionBounds[1] <= bStartY) {
          selectionBounds[3] += 1;
        } else {
          selectionBounds[1] -= 1;
        }

        padToDistribute -= 2;
      }

      return selectionBounds;
    });

    if (sketches.debug) {
      sketches.debug.clear();
      selectionBoundsBySize.forEach((b, i) => {
        if (i > 1) {
          return;
        }
        sketches.debug.noFill();
        sketches.debug.stroke(sketches.debug.color(selectionColors[i]));
        sketches.debug.rect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
      });
    }

    const selections = selectionBoundsBySize.map(selectionBounds => {
      const [ bStartX, bStartY, bEndX, bEndY ] = selectionBounds;
      return p.get(bStartX, bStartY, bEndX - bStartX, bEndY - bStartY);
    })
    return selections;
  }

  function onChange(bounds, lastPos) {
    console.log('Edit with bounds ' + bounds + ' and last position at ' + lastPos);
    // get previous state of change area
    const selections_p0 = getChangeSelections(sketches.stored, bounds, lastPos);

    // write temp to stored
    const temp = sketches.temp.get();
    sketches.stored.image(temp, 0, 0);
    sketches.temp.clear();

    // get new state of change area
    const selections_p1 = getChangeSelections(sketches.stored, bounds, lastPos);

    // render to displays and retrieve dataURLs (format for API)
    const befores = selections_p0.map((s, i) => {
      const p = sketches.selections.p0[i];
      p.background(255);
      p.image(s, 0, 0);
      const dataURL = p.canvas.toDataURL();
      return dataURL;
    });
    const afters = selections_p1.map((s, i) => {
      const p = sketches.selections.p1[i];
      p.background(255);
      p.image(s, 0, 0);
      const dataURL = p.canvas.toDataURL();
      return dataURL;
    });
    const img = sketches.stored.canvas.toDataURL();

    // find other areas of canvas that match p0
    // send canvas and img selections to have activations evaluated
    // find nearest neighbors (probably should do a 2D search tree)
    // get similar sketch segments to p1
    // keep massive hash tables of drawing sections (should serialize so we can quickly load)
    // apply p1 matches to p0 matches
    // align p1 to p0, draw using temporal agent
    const scale = getAppropriateScale(bounds);
    getSuggestions(befores, afters, img, scale);
  }

  function getAppropriateScale(bounds) {
    const change_size = Math.max(bounds[2] - bounds[0], bounds[3] - bounds[1]);
    const size_diffs = selectionSizes.map(s => Math.abs(s - change_size))
    let min_diff = Infinity;
    let min_idx = -1;
    console.log(size_diffs);
    size_diffs.forEach((diff, i) => {
      if (diff < min_diff) {
        min_diff = diff;
        min_idx = i;
      }
    });
    const scale = min_idx;
    return scale;
  }

  function getSelectionSketch(layer_i, p_i) {
    return (p) => {
      sketches.selections['p' + p_i][layer_i - 1] = p;
      p.setup = function setup() {
        const sketchSize = selectionSizes[layer_i - 1];
        p.pixelDensity(1);
        p.createCanvas(sketchSize, sketchSize);
        p.background(255);
        p.noLoop();
      };

      p.draw = function draw() {
      };
    };
  }
  new p5(getSelectionSketch(1, 0), $('#L1_matches .selection.p0')[0]);
  new p5(getSelectionSketch(1, 1), $('#L1_matches .selection.p1')[0]);
  new p5(getSelectionSketch(2, 0), $('#L2_matches .selection.p0')[0]);
  new p5(getSelectionSketch(2, 1), $('#L2_matches .selection.p1')[0]);
  new p5(getSelectionSketch(3, 0), $('#L3_matches .selection.p0')[0]);
  new p5(getSelectionSketch(3, 1), $('#L3_matches .selection.p1')[0]);
  new p5(getSelectionSketch(4, 0), $('#L4_matches .selection.p0')[0]);
  new p5(getSelectionSketch(4, 1), $('#L4_matches .selection.p1')[0]);
  new p5(getSelectionSketch(5, 0), $('#L5_matches .selection.p0')[0]);
  new p5(getSelectionSketch(5, 1), $('#L5_matches .selection.p1')[0]);

  function sketch_stored(p) {
    sketches.stored = p;

    p.setup = function setup() {
      p.pixelDensity(1);
      p.createCanvas(canvasSizeX, canvasSizeY);
      p.background(255);
    };

    p.draw = function draw() {
      p.line(0, p.height / 1.5, p.width, p.height / 1.5);
      //use esc to empty canvas
      if (p.keyIsPressed) {
        if (p.keyCode == p.ESCAPE) {
          p.background(255);
        }
      }
    };
  }
  new p5(sketch_stored, document.getElementById('sketch_stored'));

  function sketch_temp(p) {
    sketches.temp = p;
    let bounds = null;
    let lastPos = null;

    p.setup = function setup() {
      p.pixelDensity(1);
      p.createCanvas(canvasSizeX, canvasSizeY);
    };

    p.draw = function draw() {
      p.fill(0);
      p.stroke(0);
      p.strokeWeight(1.3);

      // Record mouse pressed events within canvas
      const px = p.pmouseX;
      const py = p.pmouseY;
      const x = p.mouseX;
      const y = p.mouseY;
      if (!(x < 0 || y < 0 || px < 0 || py < 0 || x >= p.width || px >= p.width || y >= p.height || py >= p.height)) {
        if (p.mouseIsPressed) {
          // draw line
          p.line(px, py, x, y);

          // update bounds of this edit
          if (!bounds) {
            let minX = Math.min(px, x);
            let minY = Math.min(py, y);
            let maxX = Math.max(px, x);
            let maxY = Math.max(py, y);
            bounds = [minX, minY, maxX, maxY];
          } else {
            bounds = [Math.min(bounds[0], x), Math.min(bounds[1], y), Math.max(bounds[2], x), Math.max(bounds[3], y)];
          }
          lastPos = [x, y];
        }
      }
      // If mouse is not pressed, and it was being pressed at the last draw, trigger on change and clear
      if (!p.mouseIsPressed) {
        if (bounds) {
          onChange(bounds, lastPos);
        }
        bounds = null;
        lastPos = null;
      }
    };
  }
  new p5(sketch_temp, document.getElementById('sketch_temp'));

  function sketch_debug(p) {
    sketches.debug = p;

    p.setup = function setup() {
      p.pixelDensity(1);
      p.createCanvas(canvasSizeX, canvasSizeY);
      p.noLoop();
    };

    p.draw = function draw() {
    };
  }
  new p5(sketch_debug, document.getElementById('sketch_debug'));

  function sketch_comp(p) {
    sketches.comp = p;

    p.setup = function setup() {
      p.pixelDensity(1);
      p.createCanvas(canvasSizeX, canvasSizeY);
      p.noLoop();
    };

    p.draw = function draw() {};
  }
  new p5(sketch_comp, document.getElementById('sketch_comp'));

  function sketch_comp2(p) {
    sketches.comp2 = p;

    p.setup = function setup() {
      p.pixelDensity(1);
      p.createCanvas(canvasSizeX, canvasSizeY);
      p.noLoop();
    };

    p.draw = function draw() {};
  }
  new p5(sketch_comp2, document.getElementById('sketch_comp2'));

  function sketch_comp_temp(p) {
    sketches.comp_temp = p;

    p.setup = function setup() {
      p.pixelDensity(1);
      p.createCanvas(canvasSizeX, canvasSizeY);
      p.noLoop();
    };

    p.draw = function draw() {};
  }
  new p5(sketch_comp_temp, document.getElementById('sketch_comp_temp'));
}());
