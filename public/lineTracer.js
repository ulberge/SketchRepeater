(function() {
  /**
   * Randomize array element order in-place.
   * Using Durstenfeld shuffle algorithm. https://stackoverflow.com/questions/2450954/how-to-randomize-shuffle-a-javascript-array
   */
  function shuffleArray(array) {
      for (var i = array.length - 1; i > 0; i--) {
          var j = Math.floor(Math.random() * (i + 1));
          var temp = array[i];
          array[i] = array[j];
          array[j] = temp;
      }
      return array;
  }

  const sketches = {};

  function sketch_img(p) {
    sketches.img = p;

    p.setup = function setup() {
      p.pixelDensity(1);
      p.createCanvas(100, 100);
      p.noLoop();
    };

    p.draw = function draw() {};
  }
  new p5(sketch_img, document.getElementById('boid_test_img'));

  function sketch_boid(p) {
    sketches.boid = p;

    p.setup = function setup() {
      p.pixelDensity(1);
      p.createCanvas(100, 100);
      p.noLoop();
    };

    p.draw = function draw() {};
  }
  new p5(sketch_boid, document.getElementById('boid_test_boid'));

  function sketch_draw(p) {
    sketches.draw = p;

    p.setup = function setup() {
      p.pixelDensity(1);
      p.createCanvas(100, 100);
      p.noLoop();
    };

    p.draw = function draw() {};
  }
  new p5(sketch_draw, document.getElementById('boid_test_draw'));

  /**
   * Modified from The Nature of Code by Daniel Shiffman
   *
   */
  function Boid(remaining) {
    // this.p = remaining;
    this.p = sketches.boid;
    this.position = null;
    this.acceleration = this.p.createVector(0, 0);
    this.velocity = this.p.createVector(0, 0);
    this.r = 1.0;
    this.maxspeed = 3;    // Maximum speed
    this.maxforce = 0.05; // Maximum steering force

    this.pixelMap = remaining.pixels;
    this.pixelMapP = remaining;
    this.dim = [remaining.width, remaining.height];

    this.done = false;
  }

  Boid.prototype.newStart = function() {
    this.done = false;
    this.position = this.findStart();
    this.acceleration = this.p.createVector(0, 0);
    this.velocity = this.p.createVector(0, 0);
    if (!this.position) {
      this.done = true;
    }
  }

  Boid.prototype.findStart = function() {
    console.log('pixels remaining: ' + this.pixelMap.filter(v => v > 0).length);

    // First check boundaries starting from random corner
    const { dim } = this;
    const outsideOptions = [
      () => { // check left row
        console.log('Check left row');
        const x = 0;
        for (let y = 0; y < dim[1]; y += 1) {
          if (this.getPixel(x, y) > 0) { return this.p.createVector(x, y) }
        }
        return null;
      },
      () => { // check right row
        console.log('Check right row');
        const x = dim[0] - 1;
        for (let y = 0; y < dim[1]; y += 1) {
          if (this.getPixel(x, y) > 0) { return this.p.createVector(x, y) }
        }
        return null;
      },
      () => { // check top row
        console.log('Check top row');
        const y = 0;
        for (let x = 0; x < dim[0]; x += 1) {
          if (this.getPixel(x, y) > 0) { return this.p.createVector(x, y) }
        }
        return null;
      },
      () => { // check bottom row
        console.log('Check bottom row');
        const y = dim[0] - 1;
        for (let x = 0; x < dim[0]; x += 1) {
          if (this.getPixel(x, y) > 0) { return this.p.createVector(x, y) }
        }
        return null;
      }
    ];

    const sweepOptions = [
      () => { // sweep from left
        console.log('Sweep from left');
        for (let x = 0; x < dim[0]; x += 1) {
          for (let y = 0; y < dim[1]; y += 1) {
            if (this.getPixel(x, y) > 0) { return this.p.createVector(x, y) }
          }
        }
        return null;
      },
      () => { // sweep from right
        console.log('Sweep from right');
        for (let x = dim[0] - 1; x >= 0; x -= 1) {
          for (let y = 0; y < dim[1]; y += 1) {
            if (this.getPixel(x, y) > 0) { return this.p.createVector(x, y) }
          }
        }
        return null;
      },
      () => { // sweep from top
        console.log('Sweep from top');
        for (let y = 0; y < dim[1]; y += 1) {
          for (let x = 0; x < dim[0]; x += 1) {
            if (this.getPixel(x, y) > 0) { return this.p.createVector(x, y) }
          }
        }
        return null;
      },
      () => { // sweep from bottom
        console.log('Sweep from bottom');
        for (let y = dim[1] - 1; y >= 0; y -= 1) {
          for (let x = 0; x < dim[0]; x += 1) {
            if (this.getPixel(x, y) > 0) { return this.p.createVector(x, y) }
          }
        }
        return null;
      }
    ];

    let order = shuffleArray([0, 1, 2, 3]);
    // let order = [0, 1, 2, 3];
    // Try outside rows for non blank
    for (let i = 0; i < 4; i += 1) {
      const pos = outsideOptions[order[i]]();
      if (pos) {
        return pos
      }
    }

    // Try sweep from certain direction
    for (let i = 0; i < 4; i += 1) {
      const pos = sweepOptions[order[i]]();
      if (pos) {
        return pos
      }
    }

    return null;
  }

  Boid.prototype.getPixel = function(x, y) {
    const index = (y * (4 * this.dim[0])) + (x * 4);
    return 255 - this.pixelMap[index];
  }

  Boid.prototype.removePixel = function(x, y) {
    const index = (4 * (y * this.dim[0])) + (x * 4);
    this.pixelMap[index] = 255;
    this.pixelMap[index+1] = 255;
    this.pixelMap[index+2] = 255;
  }

  Boid.prototype.run = function() {
    // console.log('run boid');
    // Find nearest neighbors with radial search from location
    const neighbors = this.getNeighbors(this.position, 4);

    // If all neighbors are 0, stop
    if (neighbors.filter(n => n.value !== 0).length === 0) {
      this.done = true;
      return;
    }
    // Apply force weighted by intensity of pixel
    const forces = neighbors.filter(n => n.value > 0).map(n => this.seek(n.position).mult(n.value));

    // We dont want forces to cancel each other out...
    const xSum = forces.reduce((a, f) => a + f.x, 0);
    const ySum = forces.reduce((a, f) => a + f.y, 0);
    forces.forEach(f => {
      f.x = Math.sign(xSum) === Math.sign(f.x) ? f.x : 0;
      f.y = Math.sign(ySum) === Math.sign(f.y) ? f.y : 0;
    });

    forces.forEach(f => this.applyForce(f));
    // console.log(this.position, this.velocity, this.acceleration);

    this.update();
    this.borders();
    this.render();
    // console.log(this.position, this.velocity, this.acceleration);

    // Remove neighbors left behind
    const backwards = this.velocity.normalize().mult(-1);
    const behind = this.position.add(backwards);
    const oldNeighbors = this.getNeighbors(behind, 2);
    oldNeighbors.forEach(n => {
      this.removePixel(n.position.x, n.position.y);
    });

    if (this.pixelMapP) {
      this.pixelMapP.updatePixels();
    }
  }

  Boid.prototype.getNeighbors = function(position, size=1) {
    const neighbors = [];
    const n2D = [];
    for (let dy = -size; dy <= size; dy += 1) {
      const row = [];
      for (let dx = -size; dx <= size; dx += 1) {
        const xi = Math.floor(position.x) + dx;
        const yi = Math.floor(position.y) + dy;
        if (xi >= 0 && yi >= 0 && xi < this.dim[0] && yi < this.dim[1]) {
          const value = this.getPixel(xi, yi);
          neighbors.push({
            value,
            position: this.p.createVector(xi, yi)
          });
          row.push(value);
        } else {
          row.push(0);
        }
      }
      n2D.push(row);
    }
    // console.log(JSON.stringify(n2D));
    // console.table(n2D);
    return neighbors;
  }

  Boid.prototype.applyForce = function(force) {
    // We could add mass here if we want A = F / M
    this.acceleration.add(force);
  }

  // Method to update location
  Boid.prototype.update = function() {
    // Update velocity
    this.velocity.add(this.acceleration);
    // Limit speed
    this.velocity.limit(this.maxspeed);
    this.position.add(this.velocity);
    // Reset accelertion to 0 each cycle
    this.acceleration.mult(0);
  }

  // A method that calculates and applies a steering force towards a target
  // STEER = DESIRED MINUS VELOCITY
  Boid.prototype.seek = function(target) {
    let desired = p5.Vector.sub(target,this.position);  // A vector pointing from the location to the target
    // Normalize desired and scale to maximum speed
    desired.normalize();
    desired.mult(this.maxspeed);
    // Steering = Desired minus Velocity
    let steer = p5.Vector.sub(desired,this.velocity);
    steer.limit(this.maxforce);  // Limit to maximum steering force
    return steer;
  }

  Boid.prototype.render = function() {
    // Draw a triangle rotated in the direction of velocity
    let theta = this.velocity.heading() + this.p.radians(90);
    this.p.clear();
    this.p.fill(0);
    this.p.stroke(0);
    this.p.strokeWeight(1);
    this.p.push();
    this.p.translate(this.position.x, this.position.y);
    this.p.rotate(theta);
    this.p.beginShape();
    this.p.vertex(0, -this.r * 2);
    this.p.vertex(-this.r, this.r * 2);
    this.p.vertex(this.r, this.r * 2);
    this.p.endShape(this.p.CLOSE);
    // this.p.line(0, 0, this.velocity.x * 100, this.velocity.y * 100);
    this.p.pop();
  }

  // Wraparound
  Boid.prototype.borders = function() {
    if (this.position.x < -this.r)  this.position.x = this.dim[0] + this.r;
    if (this.position.y < -this.r)  this.position.y = this.dim[1] + this.r;
    if (this.position.x > this.dim[0] + this.r) this.position.x = -this.r;
    if (this.position.y > this.dim[1] + this.r) this.position.y = -this.r;
  }

  async function pause(t) {
    return new Promise((resolve) => {
      setTimeout(() => {
        resolve();
      }, t);
    });
  }

  async function getBuffer(p, dataURL) {
    return new Promise((resolve) => {
      const raw = new Image();
      raw.src = 'data:image/jpeg;base64,' + dataURL;
      raw.onload = function() {
        const img = p.createImage(raw.width, raw.height);
        img.drawingContext.drawImage(raw, 0, 0);

        // draw to graphics
        const g = p.createGraphics(raw.width, raw.height)
        g.image(img, 0, 0);
        resolve(g);
      }
    });
  }

  async function traceOne(p, position, boid, runsRemaining, speed) {
    boid.newStart();
    if (boid.done) {
      console.log('No start found!');
      return;
    }
    console.log('Trace one starting at', boid.position);

    let previousPos = boid.position;
    for (let i = 0; i < runsRemaining; i += 1) {
      boid.run();
      const pos = boid.position;
      if (i > 2 && previousPos) {
        p.push();
        p.translate(position.x, position.y);
        p.stroke(0);
        p.strokeWeight(1.3);
        p.line(previousPos.x, previousPos.y, pos.x, pos.y);
        p.pop();

        sketches.draw.line(previousPos.x, previousPos.y, pos.x, pos.y);
      }
      previousPos = pos.copy();

      if (boid.done) {
        console.log('Boid done with trace one', boid.position);
        return runsRemaining;
      }
      sketches.img.image(boid.pixelMapP, 0, 0);
      await pause(2);
      runsRemaining -= 1;
    }
    return runsRemaining;
  }

  // Given a canvas, an image, a canvas position
  // Have a boid trace lines from the image
  async function trace(pMain, p, position, dataURL, numLines=3, speed=10) {
    const remaining = await getBuffer(p, dataURL);
    remaining.loadPixels();
    const boid = new Boid(remaining);

    sketches.img.resizeCanvas(remaining.width, remaining.height);
    sketches.boid.resizeCanvas(remaining.width, remaining.height);
    sketches.draw.resizeCanvas(remaining.width, remaining.height);

    let runsRemaining = 200;
    let tryCount = 0;
    while (runsRemaining > 0 && tryCount < numLines) {
      console.log('try count: ' + (tryCount + 1), 'runs remaining', runsRemaining);
      runsRemaining = await traceOne(p, position, boid, runsRemaining);
      tryCount += 1;
    }
    const temp = p.get();
    pMain.image(temp, 0, 0);
    p.clear();
  }

  // setTimeout(() => {
  //   trace();
  // }, 1000);

  window.lineTracer = {
    trace
  };
}());
