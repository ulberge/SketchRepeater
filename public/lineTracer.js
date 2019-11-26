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
    this.maxspeed = 2;    // Maximum speed
    this.maxforce = 2; // Maximum steering force
    this.maxacc = 2; // Maximum steering force

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
    // console.log('pixels remaining: ' + this.pixelMap.filter(v => v > 0).length);

    // First check boundaries starting from random corner
    const { dim } = this;
    const outsideOptions = [
      () => { // check left row
        // console.log('Check left row');
        const x = 0;
        for (let y = 0; y < dim[1]; y += 1) {
          if (this.getPixel(x, y) > 0) { return this.p.createVector(x, y) }
        }
        return null;
      },
      () => { // check right row
        // console.log('Check right row');
        const x = dim[0] - 1;
        for (let y = 0; y < dim[1]; y += 1) {
          if (this.getPixel(x, y) > 0) { return this.p.createVector(x, y) }
        }
        return null;
      },
      () => { // check top row
        // console.log('Check top row');
        const y = 0;
        for (let x = 0; x < dim[0]; x += 1) {
          if (this.getPixel(x, y) > 0) { return this.p.createVector(x, y) }
        }
        return null;
      },
      () => { // check bottom row
        // console.log('Check bottom row');
        const y = dim[0] - 1;
        for (let x = 0; x < dim[0]; x += 1) {
          if (this.getPixel(x, y) > 0) { return this.p.createVector(x, y) }
        }
        return null;
      }
    ];

    const sweepOptions = [
      () => { // sweep from left
        // console.log('Sweep from left');
        for (let x = 0; x < dim[0]; x += 1) {
          for (let y = 0; y < dim[1]; y += 1) {
            if (this.getPixel(x, y) > 0) { return this.p.createVector(x, y) }
          }
        }
        return null;
      },
      () => { // sweep from right
        // console.log('Sweep from right');
        for (let x = dim[0] - 1; x >= 0; x -= 1) {
          for (let y = 0; y < dim[1]; y += 1) {
            if (this.getPixel(x, y) > 0) { return this.p.createVector(x, y) }
          }
        }
        return null;
      },
      () => { // sweep from top
        // console.log('Sweep from top');
        for (let y = 0; y < dim[1]; y += 1) {
          for (let x = 0; x < dim[0]; x += 1) {
            if (this.getPixel(x, y) > 0) { return this.p.createVector(x, y) }
          }
        }
        return null;
      },
      () => { // sweep from bottom
        // console.log('Sweep from bottom');
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
    const searchSize = 8;
    // // console.log('run boid');
    // Find nearest neighbors with search forward from location
    const forwards = this.velocity.copy().normalize().mult(3);
    const ahead = this.position.copy().add(forwards);
    const neighbors = this.getNeighbors(ahead, searchSize);
    // console.table(neighbors);
    this.p.clear();

    // If all neighbors are 0, stop
    // // console.log('neigbs', neighbors.filter(n => n.value !== 0).length);
    if (neighbors.filter(n => n.value !== 0).length === 0) {
      this.done = true;
      return;
    }
    // Apply force weighted by intensity of pixel
    let forces = neighbors.filter(n => n.value > 0).map(n => n.direction.copy().mult(n.value / 255)).filter(f => f.mag() > 0);

    // We dont want forces to cancel each other out...
    // Cluster forces with previous forces within PI/4 radians
    if (forces.length > 1) {
      // start with velocity as preferred heading
      const forceGroups = [[this.velocity.copy().mult(0.1)]];
      for (let i = 0; i < forces.length; i += 1) {
        const force = forces[i];
        // check against first member of each force group
        let groupIndex = null;
        forceGroups.forEach((group, i) => {
          const leader = group[0];
          const diff = leader.angleBetween(force);
          if (Math.abs(diff) < (Math.PI / 4)) {
            groupIndex = i;
          }
        });

        if (groupIndex !== null) {
          forceGroups[groupIndex].push(force);
        } else {
          forceGroups.push([force]);
        }
      }

      forceGroups.forEach(group => {
        const start = this.p.createVector(0, 0);
        group.forEach(f => start.add(f));
        start.mult(10);
        this.p.push();
        this.p.stroke('#ff0000');
        this.p.line(this.position.x, this.position.y, this.position.x + start.x, this.position.y + start.y)
        this.p.pop();
      });

      // Choose group with greatest magnitude and apply those forces
      const groupMags = forceGroups.map(group => group.reduce((a, f) => a += f.mag(), 0));
      let maxIndex = groupMags.indexOf(Math.max(...groupMags));
      if (groupMags[0] > 1) {
        maxIndex = 0;
      }

      // Only observe forces in dominant direction
      forces = forceGroups[maxIndex];
    }

    forces.forEach(f => this.applyForce(f));
    this.acceleration.normalize().mult(this.maxacc);

    this.update();
    // this.borders();

    const newForwards = this.velocity.copy().normalize().mult(3);
    const newAhead = this.position.copy().add(newForwards);
    this.getNeighbors(newAhead, searchSize);


    // Remove neighbors left behind
    const backwards = this.velocity.copy().normalize().mult(2);
    const behind = this.position.copy().add(backwards);


    // // const behind = this.position.copy();
    // this.getNeighbors(behind, 3, 1);

    const oldNeighbors = this.getWake(behind, 6);
    oldNeighbors.forEach(n => {
      this.removePixel(n.position.x, n.position.y);
    });

    if (this.pixelMapP) {
      this.pixelMapP.updatePixels();
    }

    this.render();
  }

  Boid.prototype.getWake = function(position, size=3) {
    // Get perp behind it and erase that...
    const perp = this.velocity.copy().normalize().rotate(this.p.HALF_PI);
    const position0 = position.copy().sub(this.velocity.copy().normalize());
    const position1 = position.copy().sub(this.velocity.copy().normalize().mult(0.25));
    const position2 = position.copy().sub(this.velocity.copy().normalize().mult(0.5));
    const position3 = position.copy().sub(this.velocity.copy().normalize().mult(0.75));
    const position4 = position.copy().sub(this.velocity.copy().normalize().mult(1.25));
    const position5 = position.copy().sub(this.velocity.copy().normalize().mult(1.5));
    const position6 = position.copy().sub(this.velocity.copy().normalize().mult(1.75));
    const position7 = position.copy().sub(this.velocity.copy().normalize().mult(2));

    const neighbors = [];

    const addXY = (position, delta) => {
      const xi = Math.floor(position.x + delta.x);
      const yi = Math.floor(position.y + delta.y);
      if (xi >= 0 && yi >= 0 && xi < this.dim[0] && yi < this.dim[1]) {
        neighbors.push({
          position: this.p.createVector(xi, yi)
        });
        this.p.push();
        this.p.noStroke();
        this.p.fill('#ff00ff');
        this.p.rect(xi, yi, 1, 1);
        this.p.pop();
      }
    };

    for (let i = 0; i < size; i += 1) {
      addXY(position0, perp.copy().mult(i));
      addXY(position0, perp.copy().mult(-i));
      addXY(position1, perp.copy().mult(i));
      addXY(position1, perp.copy().mult(-i));
      addXY(position2, perp.copy().mult(i));
      addXY(position2, perp.copy().mult(-i));
      addXY(position3, perp.copy().mult(i));
      addXY(position3, perp.copy().mult(-i));
      addXY(position4, perp.copy().mult(i));
      addXY(position4, perp.copy().mult(-i));
      addXY(position5, perp.copy().mult(i));
      addXY(position5, perp.copy().mult(-i));
      addXY(position6, perp.copy().mult(i));
      addXY(position6, perp.copy().mult(-i));
      addXY(position7, perp.copy().mult(i));
      addXY(position7, perp.copy().mult(-i));
    }

    return neighbors;
  }

  /* A utility function to calculate area of triangle
  formed by (x1, y1) (x2, y2) and (x3, y3) */
  function area(x1, y1, x2, y2, x3, y3) {
    return Math.abs((x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))/2.0);
  }

  /* A function to check whether poP(x, y) lies
  inside the triangle formed by A(x1, y1),
  B(x2, y2) and C(x3, y3) */
  function isInside(x1, y1, x2, y2, x3, y3, x, y) {
    /* Calculate area of triangle ABC */
    const A = area(x1, y1, x2, y2, x3, y3);

    /* Calculate area of triangle PBC */
    const A1 = area(x, y, x2, y2, x3, y3);

    /* Calculate area of triangle PAC */
    const A2 = area(x1, y1, x, y, x3, y3);

    /* Calculate area of triangle PAB */
    const A3 = area(x1, y1, x2, y2, x, y);

    /* Check if sum of A1, A2 and A3 is same as A */
    return A === (A1 + A2 + A3);
  }

  Boid.prototype.getNeighbors = function(position, size=1, type=0) {
    const t1 = position.copy().add(this.velocity.copy().normalize().mult(size));
    const t2 = position.copy().add(this.velocity.copy().rotate(90).normalize().mult(size - 1));
    const t3 = position.copy().add(this.velocity.copy().rotate(-90).normalize().mult(size - 1));

    const neighbors = [];
    const n2D = [];
    for (let dy = -size; dy <= size; dy += 1) {
      const row = [];
      for (let dx = -size; dx <= size; dx += 1) {
        const xi = Math.floor(position.x) + dx;
        const yi = Math.floor(position.y) + dy;

        if (!isInside(Math.floor(t1.x), Math.floor(t1.y), Math.floor(t2.x), Math.floor(t2.y), Math.floor(t3.x), Math.floor(t3.y), xi, yi)) {
          continue;
        }

        if (xi >= 0 && yi >= 0 && xi < this.dim[0] && yi < this.dim[1]) {
          const value = this.getPixel(xi, yi);
          neighbors.push({
            value,
            position: this.p.createVector(xi, yi),
            direction: this.p.createVector(dx, dy)
          });
          row.push(value);
        } else {
          row.push(0);
        }

        // Draw for debug
        this.p.push();
        this.p.noStroke();
        if (type === 0) {
          this.p.fill('#ff0000');
        } else {
          this.p.fill('#0000ff');
        }
        this.p.rect(xi, yi, 1, 1);
        this.p.pop();
      }
      n2D.push(row);
    }
    // // console.log(JSON.stringify(n2D));
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

  async function getBuffer(p, dataURL, padding) {
    return new Promise((resolve) => {
      const raw = new Image();
      raw.src = 'data:image/jpeg;base64,' + dataURL;
      raw.onload = function() {
        const img = p.createImage(raw.width, raw.height);
        img.drawingContext.drawImage(raw, 0, 0);

        // draw to graphics
        const g = p.createGraphics(raw.width, raw.height)
        g.image(img, 0, 0);
        g.noStroke();
        g.fill(255);
        console.log(g.width);
        g.rect(0, 0, g.width, padding);
        g.rect(0, g.height - padding, g.width, g.height);
        g.rect(0, 0, padding, g.height);
        g.rect(g.width - padding, 0, g.width, g.height);
        resolve(g);
      }
    });
  }

  async function traceOne(p, position, boid, runsRemaining, speed) {
    boid.newStart();
    if (boid.done) {
      // console.log('No start found!');
      return;
    }
    // console.log('Trace one starting at', boid.position);

    let previousPos = boid.position;
    while (runsRemaining > 0) {
      boid.run();
      const pos = boid.position;
      if (previousPos) {
        p.push();
        p.translate(position.x, position.y);
        p.stroke('#660000');
        p.strokeWeight(1.3);
        p.line(previousPos.x, previousPos.y, pos.x, pos.y);
        p.pop();

        if (Math.abs(previousPos.x - pos.x) + Math.abs(previousPos.y - pos.y) > 30) {
          debugger;
        }

        sketches.draw.line(previousPos.x, previousPos.y, pos.x, pos.y);
      }
      previousPos = pos.copy();

      if (boid.done) {
        // console.log('Boid done with trace one', boid.position);
        return runsRemaining;
      }
      sketches.img.image(boid.pixelMapP, 0, 0);
      await pause(speed);
      runsRemaining -= 1;
    }
    return runsRemaining;
  }

  // Given a canvas, an image, a canvas position
  // Have a boid trace lines from the image
  async function trace(p, position, dataURL, numLines=3, speed=10, padding=0) {
    const remaining = await getBuffer(p, dataURL, padding);
    remaining.loadPixels();
    const boid = new Boid(remaining);

    sketches.img.resizeCanvas(remaining.width, remaining.height);
    sketches.boid.resizeCanvas(remaining.width, remaining.height);
    sketches.draw.resizeCanvas(remaining.width, remaining.height);

    let runsRemaining = 400;
    let tryCount = 0;
    while (runsRemaining > 0 && tryCount < numLines) {
      // console.log('try count: ' + (tryCount + 1), 'runs remaining', runsRemaining);
      runsRemaining = await traceOne(p, position, boid, runsRemaining, speed);
      tryCount += 1;
    }
    console.log('end tracing, remaining: ' + runsRemaining);
  }

  // setTimeout(() => {
  //   trace();
  // }, 1000);

  window.lineTracer = {
    trace
  };
}());
