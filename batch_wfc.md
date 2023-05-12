# batch wfc (AC3)
* init possibilites, propagators
* undecided = list of indices
* main loop:
    * decide batch size, n
    * indices = choose n indices, possibly unnear each other
    * for indices: pick a random tile from possibiliites
    * for indices: update possibilites.
    * set changedTiles = list of changed tiles
    * while changedTiles > 0:
        * set recheckList = all neighbour tiles + dir pairs
        * for recheckList, for pattern: find patterns with no support
        * update undecided
        * ban patterns, set changedTiles