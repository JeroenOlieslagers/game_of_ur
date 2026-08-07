use std::cmp::Ordering;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};
use std::thread;
use std::time::Instant;

const WIDTH: usize = 3;
const HEIGHT: usize = 8;
const BOARD_LEN: usize = WIDTH * HEIGHT;
const ROSETTES: [usize; 5] = [0, 2, 10, 18, 20];
const LOOKUP_PREFIX_BITS: usize = 20;
const LOOKUP_PREFIX_COUNT: usize = 1 << LOOKUP_PREFIX_BITS;
const LOOKUP_LOWER_BITS: usize = 32 - LOOKUP_PREFIX_BITS;

// Exact on-board paths, derived from the waypoints in RoyalUr-Java's
// MastersPathPair.java and BellPathPair.java with the off-board start and end
// tiles removed. Board indices are (y - 1) * WIDTH + (x - 1) for the 1-based
// (x, y) tile coordinates used there.
const MASTERS_LIGHT_PATH: [usize; 16] = [9, 6, 3, 0, 1, 4, 7, 10, 13, 16, 19, 20, 23, 22, 21, 18];
const MASTERS_DARK_PATH: [usize; 16] = [11, 8, 5, 2, 1, 4, 7, 10, 13, 16, 19, 18, 21, 22, 23, 20];
// Bell is the Finkel path: straight up the near column, down the middle, back
// up the near column, without the Masters detour through the far column.
const BELL_LIGHT_PATH: [usize; 14] = [9, 6, 3, 0, 1, 4, 7, 10, 13, 16, 19, 22, 21, 18];
const BELL_DARK_PATH: [usize; 14] = [11, 8, 5, 2, 1, 4, 7, 10, 13, 16, 19, 22, 23, 20];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RuleSet {
    Blitz,
    Masters,
    Finkel,
}

impl RuleSet {
    fn name(self) -> &'static str {
        match self {
            Self::Blitz => "blitz",
            Self::Masters => "masters",
            Self::Finkel => "finkel",
        }
    }

    fn pieces(self) -> u8 {
        match self {
            Self::Blitz => 5,
            Self::Masters | Self::Finkel => 7,
        }
    }

    fn captures_grant_roll(self) -> bool {
        matches!(self, Self::Blitz)
    }

    /// Finkel protects pieces standing on a rosette from capture.
    fn safe_rosettes(self) -> bool {
        matches!(self, Self::Finkel)
    }

    fn light_path(self) -> &'static [usize] {
        match self {
            Self::Blitz | Self::Masters => &MASTERS_LIGHT_PATH,
            Self::Finkel => &BELL_LIGHT_PATH,
        }
    }

    fn dark_path(self) -> &'static [usize] {
        match self {
            Self::Blitz | Self::Masters => &MASTERS_DARK_PATH,
            Self::Finkel => &BELL_DARK_PATH,
        }
    }

    fn roll(self, rng: &mut SplitMix64) -> u8 {
        match self {
            // Four binary dice.
            Self::Blitz | Self::Finkel => (rng.next_u64() as u8 & 0x0f).count_ones() as u8,
            // Three binary dice, where a roll of zero counts as four.
            Self::Masters => {
                let count = (rng.next_u64() as u8 & 0x07).count_ones() as u8;
                if count == 0 { 4 } else { count }
            }
        }
    }

    fn possible_rolls(self) -> &'static [u8] {
        match self {
            Self::Blitz | Self::Finkel => &[0, 1, 2, 3, 4],
            Self::Masters => &[1, 2, 3, 4],
        }
    }

    fn roll_probabilities(self) -> &'static [f64; 5] {
        match self {
            Self::Blitz | Self::Finkel => {
                &[1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0]
            }
            Self::Masters => &[0.0, 3.0 / 8.0, 3.0 / 8.0, 1.0 / 8.0, 1.0 / 8.0],
        }
    }
}

#[derive(Clone, Debug)]
struct Game {
    rules: RuleSet,
    board: [i8; BOARD_LEN],
    light_pieces: u8,
    dark_pieces: u8,
    light_score: u8,
    dark_score: u8,
    is_light_turn: bool,
    roll: i8,
    finished: bool,
}

impl Game {
    fn initial(rules: RuleSet) -> Self {
        Self {
            rules,
            board: [0; BOARD_LEN],
            light_pieces: rules.pieces(),
            dark_pieces: rules.pieces(),
            light_score: 0,
            dark_score: 0,
            is_light_turn: true,
            roll: -1,
            finished: false,
        }
    }

    fn path(&self) -> &'static [usize] {
        if self.is_light_turn { self.rules.light_path() } else { self.rules.dark_path() }
    }

    fn turn_sign(&self) -> i8 {
        if self.is_light_turn { 1 } else { -1 }
    }

    fn turn_pieces(&self) -> u8 {
        if self.is_light_turn { self.light_pieces } else { self.dark_pieces }
    }

    fn available_moves(&self, output: &mut [i8; 8]) -> usize {
        let roll = self.roll;
        assert!(roll >= 0);
        let roll = roll as usize;
        let path = self.path();
        let sign = self.turn_sign();
        let mut count = 0usize;

        if roll <= path.len() {
            let source = path.len() - roll;
            if self.board[path[source]] == sign * (source as i8 + 1) {
                output[count] = source as i8;
                count += 1;
            }
        }

        let max_source_exclusive = path.len().saturating_sub(roll);
        for encoded_source in 0..=max_source_exclusive {
            let source: i8 = encoded_source as i8 - 1;
            if source >= 0 {
                let source_index = source as usize;
                if self.board[path[source_index]] != sign * (source + 1) {
                    continue;
                }
            } else if self.turn_pieces() == 0 {
                continue;
            }

            let destination_path = (source as isize + roll as isize) as usize;
            if destination_path >= path.len() {
                continue;
            }
            let destination_tile = path[destination_path];
            let destination_piece = self.board[destination_tile];
            if destination_piece != 0 {
                // Can't land on your own piece.
                if destination_piece * sign > 0 {
                    continue;
                }
                // Can't capture an opposing piece standing on a rosette when
                // rosettes are safe (Finkel).
                if self.rules.safe_rosettes() && ROSETTES.contains(&destination_tile) {
                    continue;
                }
            }
            output[count] = source;
            count += 1;
        }
        count
    }

    fn apply_roll(&mut self, roll: u8, moves: &mut [i8; 8]) -> usize {
        assert!(self.roll < 0);
        if roll == 0 {
            self.is_light_turn = !self.is_light_turn;
            return 0;
        }
        self.roll = roll as i8;
        let count = self.available_moves(moves);
        if count == 0 {
            self.is_light_turn = !self.is_light_turn;
            self.roll = -1;
        }
        count
    }

    fn apply_move(&mut self, source: i8, rules: RuleSet) {
        let roll = self.roll;
        assert!(roll > 0);
        self.roll = -1;
        let path = self.path();
        let sign = self.turn_sign();

        if source >= 0 {
            self.board[path[source as usize]] = 0;
        } else if self.is_light_turn {
            self.light_pieces -= 1;
        } else {
            self.dark_pieces -= 1;
        }

        let destination_path = source as isize + roll as isize;
        if destination_path < path.len() as isize {
            let destination_path = destination_path as usize;
            let destination = path[destination_path];
            let captured = self.board[destination];
            if captured > 0 {
                self.light_pieces += 1;
            } else if captured < 0 {
                self.dark_pieces += 1;
            }
            self.board[destination] = sign * (destination_path as i8 + 1);

            let extra = ROSETTES.contains(&destination)
                || (rules.captures_grant_roll() && captured != 0);
            if !extra {
                self.is_light_turn = !self.is_light_turn;
            }
        } else {
            if self.is_light_turn {
                self.light_score += 1;
                if self.light_score >= rules.pieces() {
                    self.finished = true;
                    return;
                }
            } else {
                self.dark_score += 1;
                if self.dark_score >= rules.pieces() {
                    self.finished = true;
                    return;
                }
            }
            self.is_light_turn = !self.is_light_turn;
        }
    }

    fn reverse_players(&self) -> Self {
        let mut result = Self {
            rules: self.rules,
            board: [0; BOARD_LEN],
            light_pieces: self.dark_pieces,
            dark_pieces: self.light_pieces,
            light_score: self.dark_score,
            dark_score: self.light_score,
            is_light_turn: !self.is_light_turn,
            roll: self.roll,
            finished: self.finished,
        };
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                result.board[(WIDTH - 1 - x) + WIDTH * y] = -self.board[x + WIDTH * y];
            }
        }
        result
    }
}

/// Identify the rule set from a map's embedded `game_settings` metadata, and
/// check the rest of the settings match what this solver implements for it.
fn detect_ruleset(metadata: &str) -> RuleSet {
    let lower = metadata.to_lowercase();
    let rules = if lower.contains("\"paths\":\"bell\"") {
        RuleSet::Finkel
    } else if lower.contains("three_binary_0eq4") {
        RuleSet::Masters
    } else if lower.contains("\"start_pieces\":5") {
        RuleSet::Blitz
    } else {
        panic!("unsupported LUT metadata: {metadata}");
    };

    // Guard against a map whose settings differ from the rules we would apply.
    let expect = |field: &str, value: bool| {
        let needle = format!("\"{field}\":{value}");
        assert!(
            lower.contains(&needle),
            "{} map metadata does not contain {needle}: {metadata}",
            rules.name()
        );
    };
    expect("safe_rosettes", rules.safe_rosettes());
    expect("captures_grant_rolls", rules.captures_grant_roll());
    expect("rosettes_grant_rolls", true);
    assert!(
        lower.contains(&format!("\"start_pieces\":{}", rules.pieces())),
        "{} map metadata does not declare start_pieces {}: {metadata}",
        rules.name(),
        rules.pieces()
    );
    rules
}

#[derive(Clone)]
struct Encoding {
    rules: RuleSet,
    war_indices: Vec<usize>,
    light_safe_indices: Vec<usize>,
    dark_safe_indices: Vec<usize>,
    light_path_indices: [i8; BOARD_LEN],
    dark_path_indices: [i8; BOARD_LEN],
    compression: Vec<i16>,
    decompression: Vec<u16>,
    decompression_counts: Vec<(u8, u8)>,
    segment_bits: usize,
    /// War tiles per compression segment.
    segment_tiles: usize,
    /// Number of war-tile segments.
    segment_count: usize,
    board_bits: usize,
}

impl Encoding {
    /// Mirrors `estimateGoodWarTileCompressionTileCount` in RoyalUr-Java's
    /// SimpleGameStateEncoding: up to 8 war tiles go in one segment, otherwise
    /// split into the fewest segments of at most 8 tiles each.
    fn segment_tile_count(war_tile_count: usize) -> usize {
        if war_tile_count <= 8 {
            return war_tile_count;
        }
        let mut segments = 2;
        let mut tile_count;
        loop {
            tile_count = (war_tile_count + segments - 1) / segments;
            segments += 1;
            if tile_count <= 8 {
                break;
            }
        }
        tile_count
    }

    fn new(rules: RuleSet) -> Self {
        let mut light_path_indices = [-1i8; BOARD_LEN];
        let mut dark_path_indices = [-1i8; BOARD_LEN];
        for (index, &tile) in rules.light_path().iter().enumerate() {
            light_path_indices[tile] = index as i8;
        }
        for (index, &tile) in rules.dark_path().iter().enumerate() {
            dark_path_indices[tile] = index as i8;
        }

        let mut war_indices = Vec::new();
        let mut light_safe_indices = Vec::new();
        let mut dark_safe_indices = Vec::new();
        for tile in 0..BOARD_LEN {
            let light = light_path_indices[tile] >= 0;
            let dark = dark_path_indices[tile] >= 0;
            match (light, dark) {
                (true, true) => war_indices.push(tile),
                (true, false) => light_safe_indices.push(tile),
                (false, true) => dark_safe_indices.push(tile),
                _ => {}
            }
        }
        // The standard board has 20 tiles; how they split between shared "war"
        // tiles and each player's private tiles depends on the path pair.
        assert_eq!(
            war_indices.len() + light_safe_indices.len() + dark_safe_indices.len(),
            20
        );
        assert_eq!(light_safe_indices.len(), dark_safe_indices.len());

        let segment_tiles = Self::segment_tile_count(war_indices.len());
        let segment_count = (war_indices.len() + segment_tiles - 1) / segment_tiles;
        // Partial trailing segments would need the encoder to shift in fewer
        // tiles for the last segment. Every supported rule set divides evenly.
        assert_eq!(
            segment_count * segment_tiles,
            war_indices.len(),
            "war tile count {} does not divide into {segment_count} segments of {segment_tiles}",
            war_indices.len()
        );

        let (compression, decompression) = Self::build_compression(rules.pieces(), segment_tiles);
        let decompression_counts = decompression
            .iter()
            .map(|&raw| {
                let mut light = 0u8;
                let mut dark = 0u8;
                for local in 0..segment_tiles {
                    match (raw >> (2 * (segment_tiles - 1 - local))) & 0x3 {
                        1 => dark += 1,
                        2 => light += 1,
                        _ => {}
                    }
                }
                (light, dark)
            })
            .collect::<Vec<_>>();
        let max_compressed = decompression.len() - 1;
        let mut segment_bits = 1usize;
        while max_compressed > (1usize << segment_bits) {
            segment_bits += 1;
        }
        let board_bits = 2 * light_safe_indices.len() + segment_count * segment_bits;
        // Cross-check against the layouts RoyalUr-Java produces, which set the
        // key widths in the published maps.
        match rules {
            RuleSet::Blitz | RuleSet::Masters => {
                assert_eq!((war_indices.len(), segment_count, segment_bits, board_bits), (12, 2, 10, 28));
            }
            RuleSet::Finkel => {
                assert_eq!((war_indices.len(), segment_count, segment_bits, board_bits), (8, 1, 13, 25));
            }
        }

        Self {
            rules,
            war_indices,
            light_safe_indices,
            dark_safe_indices,
            light_path_indices,
            dark_path_indices,
            compression,
            decompression,
            decompression_counts,
            segment_bits,
            segment_tiles,
            segment_count,
            board_bits,
        }
    }

    fn build_compression(pieces: u8, tile_count: usize) -> (Vec<i16>, Vec<u16>) {
        fn visit(
            light: i8,
            dark: i8,
            state: u16,
            remaining: usize,
            values: &mut Vec<u16>,
        ) {
            for occupant in 0..3u16 {
                let mut new_light = light;
                let mut new_dark = dark;
                if occupant == 1 {
                    new_dark -= 1;
                } else if occupant == 2 {
                    new_light -= 1;
                }
                if new_light < 0 || new_dark < 0 {
                    continue;
                }
                let new_state = (state << 2) | occupant;
                if remaining == 1 {
                    values.push(new_state);
                } else {
                    visit(new_light, new_dark, new_state, remaining - 1, values);
                }
            }
        }

        let mut decompression = Vec::new();
        visit(pieces as i8, pieces as i8, 0, tile_count, &mut decompression);
        let mut compression = vec![-1i16; 1usize << (2 * tile_count)];
        for (index, &state) in decompression.iter().enumerate() {
            compression[state as usize] = index as i16;
        }
        (compression, decompression)
    }

    fn encode_light_turn(&self, game: &Game) -> u64 {
        assert!(game.is_light_turn);
        let safe_bits = self.light_safe_indices.len();
        let war_bits = self.segment_count * self.segment_bits;

        let mut dark_safe = 0u64;
        for (index, &tile) in self.dark_safe_indices.iter().enumerate() {
            if game.board[tile] != 0 {
                dark_safe |= 1u64 << index;
            }
        }
        let mut light_safe = 0u64;
        for (index, &tile) in self.light_safe_indices.iter().enumerate() {
            if game.board[tile] != 0 {
                light_safe |= 1u64 << index;
            }
        }

        let mut war = 0u64;
        for segment in 0..self.segment_count {
            let mut raw = 0usize;
            for local in 0..self.segment_tiles {
                let piece = game.board[self.war_indices[segment * self.segment_tiles + local]];
                let occupant = if piece == 0 { 0 } else if piece < 0 { 1 } else { 2 };
                raw = (raw << 2) | occupant;
            }
            let compressed = self.compression[raw];
            assert!(compressed >= 0);
            war = (war << self.segment_bits) | compressed as u64;
        }

        let board = dark_safe | (war << safe_bits) | (light_safe << (safe_bits + war_bits));
        board
            | ((game.dark_pieces as u64) << self.board_bits)
            | ((game.light_pieces as u64) << (self.board_bits + 3))
    }

    fn encode_symmetrical(&self, game: &Game) -> u64 {
        if game.is_light_turn {
            self.encode_light_turn(game)
        } else {
            self.encode_light_turn(&game.reverse_players())
        }
    }

    fn decode(&self, key: u64) -> Game {
        let safe_bits = self.light_safe_indices.len();
        let war_mask = (1u64 << self.segment_bits) - 1;
        let board_mask = (1u64 << self.board_bits) - 1;
        let board_code = key & board_mask;
        let war_bits = self.segment_count * self.segment_bits;
        let dark_safe = board_code & ((1u64 << safe_bits) - 1);
        let war = (board_code >> safe_bits) & ((1u64 << war_bits) - 1);
        let light_safe = board_code >> (safe_bits + war_bits);
        let dark_pieces = ((key >> self.board_bits) & 0x7) as u8;
        let light_pieces = ((key >> (self.board_bits + 3)) & 0x7) as u8;

        let mut game = Game {
            rules: self.rules,
            board: [0; BOARD_LEN],
            light_pieces,
            dark_pieces,
            light_score: 0,
            dark_score: 0,
            is_light_turn: true,
            roll: -1,
            finished: false,
        };

        for (index, &tile) in self.dark_safe_indices.iter().enumerate() {
            if ((dark_safe >> index) & 1) != 0 {
                game.board[tile] = -(self.dark_path_indices[tile] + 1);
            }
        }
        for (index, &tile) in self.light_safe_indices.iter().enumerate() {
            if ((light_safe >> index) & 1) != 0 {
                game.board[tile] = self.light_path_indices[tile] + 1;
            }
        }
        for segment in 0..self.segment_count {
            let shift = (self.segment_count - 1 - segment) * self.segment_bits;
            let compressed = ((war >> shift) & war_mask) as usize;
            assert!(compressed < self.decompression.len());
            let raw = self.decompression[compressed];
            for local in 0..self.segment_tiles {
                let occupant_shift = 2 * (self.segment_tiles - 1 - local);
                let occupant = (raw >> occupant_shift) & 0x3;
                let tile = self.war_indices[segment * self.segment_tiles + local];
                game.board[tile] = match occupant {
                    0 => 0,
                    1 => -(self.dark_path_indices[tile] + 1),
                    2 => self.light_path_indices[tile] + 1,
                    _ => unreachable!(),
                };
            }
        }

        let light_on_board = game.board.iter().filter(|&&piece| piece > 0).count() as u8;
        let dark_on_board = game.board.iter().filter(|&&piece| piece < 0).count() as u8;
        game.light_score = self.rules.pieces() - game.light_pieces - light_on_board;
        game.dark_score = self.rules.pieces() - game.dark_pieces - dark_on_board;
        game.finished = game.light_score >= self.rules.pieces();
        game
    }

    #[inline]
    fn scores(&self, key: u64) -> (u8, u8) {
        let safe_bits = self.light_safe_indices.len();
        let war_mask = (1u64 << self.segment_bits) - 1;
        let board_mask = (1u64 << self.board_bits) - 1;
        let war_bits = self.segment_count * self.segment_bits;
        let board = key & board_mask;
        let dark_safe = board & ((1u64 << safe_bits) - 1);
        let war = (board >> safe_bits) & ((1u64 << war_bits) - 1);
        let light_safe = board >> (safe_bits + war_bits);

        let mut light_on_board = light_safe.count_ones() as u8;
        let mut dark_on_board = dark_safe.count_ones() as u8;
        for segment in 0..self.segment_count {
            let shift = (self.segment_count - 1 - segment) * self.segment_bits;
            let compressed = ((war >> shift) & war_mask) as usize;
            let (light, dark) = self.decompression_counts[compressed];
            light_on_board += light;
            dark_on_board += dark;
        }

        let dark_pieces = ((key >> self.board_bits) & 0x7) as u8;
        let light_pieces = ((key >> (self.board_bits + 3)) & 0x7) as u8;
        (
            self.rules.pieces() - light_pieces - light_on_board,
            self.rules.pieces() - dark_pieces - dark_on_board,
        )
    }
}

#[derive(Clone, Copy, Debug)]
struct MapLayout {
    count: usize,
    key_offset: usize,
    value_offset: usize,
}

struct Lut {
    data: Vec<u8>,
    maps: Vec<MapLayout>,
    prefix_starts: Vec<Vec<u32>>,
    map_bases: Vec<usize>,
    total: usize,
    value_bytes: usize,
    rules: RuleSet,
    encoding: Encoding,
    metadata: String,
}

impl Lut {
    fn read(path: &Path) -> Self {
        let started = Instant::now();
        let data = fs::read(path).expect("failed to read LUT");
        assert_eq!(&data[0..4], b"RGU\0");
        let metadata_len = read_u32(&data, 4) as usize;
        let metadata = String::from_utf8(data[8..8 + metadata_len].to_vec()).unwrap();
        let metadata_lower = metadata.to_lowercase();
        let value_bytes = if metadata_lower.contains("\"value_type\":\"f64\"") { 8 } else { 2 };
        let rules = detect_ruleset(&metadata);

        let mut cursor = 8 + metadata_len;
        let map_count = read_u32(&data, cursor) as usize;
        cursor += 4;
        let mut counts = Vec::with_capacity(map_count);
        for _ in 0..map_count {
            counts.push(read_u32(&data, cursor) as usize);
            cursor += 4;
        }
        let total = counts.iter().sum::<usize>();
        let mut key_cursor = cursor;
        let mut value_cursor = cursor + 4 * total;
        let mut maps = Vec::with_capacity(map_count);
        let mut map_bases = Vec::with_capacity(map_count);
        let mut global_base = 0usize;
        for &count in &counts {
            map_bases.push(global_base);
            maps.push(MapLayout { count, key_offset: key_cursor, value_offset: value_cursor });
            key_cursor += 4 * count;
            value_cursor += value_bytes * count;
            global_base += count;
        }
        assert_eq!(value_cursor, data.len());

        // Restrict each binary search to keys sharing the same top 20 bits.
        // This keeps random lookups fast even in the 501-million-entry Masters map.
        let mut prefix_starts = Vec::with_capacity(map_count);
        for map in &maps {
            let mut starts = vec![0u32; LOOKUP_PREFIX_COUNT + 1];
            let mut index = 0usize;
            for prefix in 0..=LOOKUP_PREFIX_COUNT {
                while index < map.count
                    && ((read_u32(&data, map.key_offset + 4 * index) as usize) >> LOOKUP_LOWER_BITS) < prefix
                {
                    index += 1;
                }
                starts[prefix] = index as u32;
            }
            prefix_starts.push(starts);
        }
        eprintln!(
            "loaded {}: {:.3} GB, {} entries, {:.2}s",
            rules.name(),
            data.len() as f64 / 1e9,
            total,
            started.elapsed().as_secs_f64()
        );
        Self { data, maps, prefix_starts, map_bases, total, value_bytes, rules, encoding: Encoding::new(rules), metadata }
    }

    fn key_at(&self, upper: usize, index: usize) -> u64 {
        let map = self.maps[upper];
        assert!(index < map.count);
        ((upper as u64) << 32) | read_u32(&self.data, map.key_offset + 4 * index) as u64
    }

    fn value_at(&self, upper: usize, index: usize) -> f64 {
        let map = self.maps[upper];
        if self.value_bytes == 8 {
            read_f64(&self.data, map.value_offset + 8 * index)
        } else {
            read_u16(&self.data, map.value_offset + 2 * index) as f64 * 100.0 / 65535.0
        }
    }

    fn key_value_at_global(&self, mut index: usize) -> (u64, f64) {
        for upper in 0..self.maps.len() {
            if index < self.maps[upper].count {
                return (self.key_at(upper, index), self.value_at(upper, index));
            }
            index -= self.maps[upper].count;
        }
        panic!("global LUT index out of bounds");
    }

    fn lookup_key(&self, key: u64) -> f64 {
        self.value_at_global(self.lookup_index(key))
    }

    fn lookup_index(&self, key: u64) -> usize {
        let upper = (key >> 32) as usize;
        assert!(upper < self.maps.len(), "upper key missing: {upper}");
        let lower = key as u32;
        let map = self.maps[upper];
        let prefix = (lower >> LOOKUP_LOWER_BITS) as usize;
        let mut lo = self.prefix_starts[upper][prefix] as usize;
        let mut hi = self.prefix_starts[upper][prefix + 1] as usize;
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let candidate = read_u32(&self.data, map.key_offset + 4 * mid);
            match candidate.cmp(&lower) {
                Ordering::Less => lo = mid + 1,
                Ordering::Greater => hi = mid,
                Ordering::Equal => return self.map_bases[upper] + mid,
            }
        }
        panic!("key missing from LUT: {key:#x}");
    }

    fn value_at_global(&self, mut index: usize) -> f64 {
        for upper in 0..self.maps.len() {
            if index < self.maps[upper].count {
                return self.value_at(upper, index);
            }
            index -= self.maps[upper].count;
        }
        panic!("global LUT value index out of bounds");
    }

    fn light_win_percent(&self, game: &Game) -> f64 {
        if game.finished {
            return if game.light_score >= self.rules.pieces() { 100.0 } else { 0.0 };
        }
        let value = self.lookup_key(self.encoding.encode_symmetrical(game));
        if game.is_light_turn { value } else { 100.0 - value }
    }
}

fn read_u32(data: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap())
}

fn read_u16(data: &[u8], offset: usize) -> u16 {
    u16::from_be_bytes(data[offset..offset + 2].try_into().unwrap())
}

fn read_f64(data: &[u8], offset: usize) -> f64 {
    f64::from_be_bytes(data[offset..offset + 8].try_into().unwrap())
}

#[derive(Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self { Self(seed) }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    fn index(&mut self, length: usize) -> usize {
        (self.next_u64() as usize) % length
    }
}

fn choose_optimal_move(lut: &Lut, game: &Game, moves: &[i8]) -> i8 {
    assert!(!moves.is_empty());
    if moves.len() == 1 {
        return moves[0];
    }
    let light_turn = game.is_light_turn;
    let mut best_move = moves[0];
    let mut best = if light_turn { f64::NEG_INFINITY } else { f64::INFINITY };
    for &source in moves {
        let mut next = game.clone();
        next.apply_move(source, lut.rules);
        let value = lut.light_win_percent(&next);
        let better = if light_turn { value > best } else { value < best };
        if better {
            best = value;
            best_move = source;
        }
    }
    best_move
}

fn play_game(
    lut: &Lut,
    mut game: Game,
    rng: &mut SplitMix64,
    epsilon_percent: u8,
    optimal_is_light: Option<bool>,
) -> bool {
    let mut move_buffer = [0i8; 8];
    let mut plies = 0usize;
    while !game.finished {
        plies += 1;
        assert!(plies < 100_000, "game failed to terminate");
        let roll = lut.rules.roll(rng);
        let move_count = game.apply_roll(roll, &mut move_buffer);
        if move_count == 0 {
            continue;
        }

        let current_is_optimal = optimal_is_light.map(|side| side == game.is_light_turn).unwrap_or(true);
        let random_move = !current_is_optimal
            && (rng.next_u64() % 10_000) < epsilon_percent as u64 * 100;
        let source = if random_move {
            move_buffer[rng.index(move_count)]
        } else {
            choose_optimal_move(lut, &game, &move_buffer[..move_count])
        };
        game.apply_move(source, lut.rules);
    }
    game.light_score >= lut.rules.pieces()
}

fn parallel_games(
    lut: &Lut,
    start_game: &Game,
    games: usize,
    seed: u64,
    epsilon_percent: u8,
    alternate_optimal_side: bool,
) -> usize {
    let threads = thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(games.max(1));
    thread::scope(|scope| {
        let mut handles = Vec::with_capacity(threads);
        let mut offset = 0usize;
        for thread_index in 0..threads {
            let count = games / threads + usize::from(thread_index < games % threads);
            let first_game = offset;
            offset += count;
            let local_start = start_game.clone();
            handles.push(scope.spawn(move || {
                let mut rng = SplitMix64::new(
                    seed ^ (thread_index as u64).wrapping_mul(0xd6e8feb86659fd93),
                );
                let mut wins = 0usize;
                for local_index in 0..count {
                    if alternate_optimal_side {
                        let optimal_is_light = (first_game + local_index) % 2 == 0;
                        let light_won = play_game(
                            lut,
                            local_start.clone(),
                            &mut rng,
                            epsilon_percent,
                            Some(optimal_is_light),
                        );
                        wins += (light_won == optimal_is_light) as usize;
                    } else {
                        wins += play_game(
                            lut,
                            local_start.clone(),
                            &mut rng,
                            epsilon_percent,
                            None,
                        ) as usize;
                    }
                }
                wins
            }));
        }
        handles.into_iter().map(|handle| handle.join().unwrap()).sum()
    })
}

fn verify(lut: &Lut, samples: usize) {
    println!("ruleset={}", lut.rules.name());
    println!("metadata={}", lut.metadata);
    println!("entries={}", lut.total);
    println!("maps={:?}", lut.maps.iter().map(|m| m.count).collect::<Vec<_>>());
    let initial = Game::initial(lut.rules);
    let initial_key = lut.encoding.encode_light_turn(&initial);
    println!("initial_key={initial_key:#x}");
    println!("initial_light_win_percent={:.10}", lut.lookup_key(initial_key));

    let mut rng = SplitMix64::new(0x5eed_1234_abcd_9876);
    let mut checked = 0usize;
    let mut transition_checked = 0usize;
    let mut moves = [0i8; 8];
    while checked < samples {
        let global = rng.index(lut.total);
        let (key, stored) = lut.key_value_at_global(global);
        let game = lut.encoding.decode(key);
        let (light_score, dark_score) = lut.encoding.scores(key);
        assert_eq!((light_score, dark_score), (game.light_score, game.dark_score), "score decoding mismatch");
        let encoded = lut.encoding.encode_light_turn(&game);
        assert_eq!(encoded, key, "encoding roundtrip failed");
        let looked_up = lut.lookup_key(key);
        assert!((stored - looked_up).abs() < 1e-12);
        checked += 1;

        if game.finished {
            continue;
        }
        for &roll in lut.rules.possible_rolls() {
            let mut rolled = game.clone();
            let count = rolled.apply_roll(roll, &mut moves);
            if count == 0 {
                if !rolled.finished {
                    let _ = lut.light_win_percent(&rolled);
                    transition_checked += 1;
                }
            } else {
                for &source in &moves[..count] {
                    let mut next = rolled.clone();
                    next.apply_move(source, lut.rules);
                    let _ = lut.light_win_percent(&next);
                    transition_checked += 1;
                }
            }
        }
    }
    println!("roundtrips_checked={checked}");
    println!("transitions_checked={transition_checked}");
}

fn write_gap_sample(lut: &Lut, path: &Path, state_samples: usize, seed: u64) {
    let mut output = BufWriter::new(File::create(path).unwrap());
    writeln!(output, "sample,key,roll,move_count,best_pct,second_pct,gap_pct,tie").unwrap();
    let mut rng = SplitMix64::new(seed);
    let mut moves = [0i8; 8];
    let mut rows = 0usize;
    let mut states = 0usize;
    let started = Instant::now();
    while states < state_samples {
        let (key, _) = lut.key_value_at_global(rng.index(lut.total));
        let game = lut.encoding.decode(key);
        if game.finished {
            continue;
        }
        states += 1;
        for &roll in lut.rules.possible_rolls() {
            let mut rolled = game.clone();
            let count = rolled.apply_roll(roll, &mut moves);
            if count <= 1 {
                continue;
            }
            let mut best = f64::NEG_INFINITY;
            let mut second = f64::NEG_INFINITY;
            for &source in &moves[..count] {
                let mut next = rolled.clone();
                next.apply_move(source, lut.rules);
                let value = lut.light_win_percent(&next);
                if value > best {
                    second = best;
                    best = value;
                } else if value > second {
                    second = value;
                }
            }
            let gap = best - second;
            writeln!(
                output,
                "{states},{key:#x},{roll},{count},{best:.12},{second:.12},{gap:.12},{}",
                gap == 0.0
            ).unwrap();
            rows += 1;
        }
        if states % 100_000 == 0 {
            eprintln!("gap states={states}/{state_samples} rows={rows} elapsed={:.1}s", started.elapsed().as_secs_f64());
        }
    }
    eprintln!("gap complete states={states} rows={rows} elapsed={:.1}s", started.elapsed().as_secs_f64());
}

fn write_compare(
    lut: &Lut,
    path: &Path,
    state_count: usize,
    games_per_state: usize,
    seed: u64,
) {
    let mut output = BufWriter::new(File::create(path).unwrap());
    writeln!(output, "state,key,predicted_pct,games,light_wins,simulated_pct").unwrap();
    let mut sampling_rng = SplitMix64::new(seed);
    let started = Instant::now();
    for state_index in 0..state_count {
        let (key, predicted) = loop {
            let candidate = lut.key_value_at_global(sampling_rng.index(lut.total));
            if !lut.encoding.decode(candidate.0).finished {
                break candidate;
            }
        };
        let start_game = lut.encoding.decode(key);
        let game_seed = seed ^ key ^ (state_index as u64).wrapping_mul(0x9e3779b97f4a7c15);
        let wins = parallel_games(lut, &start_game, games_per_state, game_seed, 0, false);
        let simulated = 100.0 * wins as f64 / games_per_state as f64;
        writeln!(output, "{state_index},{key:#x},{predicted:.12},{games_per_state},{wins},{simulated:.12}").unwrap();
        eprintln!(
            "compare state={}/{} predicted={:.4}% simulated={:.4}% elapsed={:.1}s",
            state_index + 1,
            state_count,
            predicted,
            simulated,
            started.elapsed().as_secs_f64()
        );
    }
}

fn write_epsilon(lut: &Lut, path: &Path, games_per_epsilon: usize, seed: u64) {
    let mut output = BufWriter::new(File::create(path).unwrap());
    writeln!(output, "epsilon,games,optimal_wins,optimal_win_pct").unwrap();
    let started = Instant::now();
    for epsilon in 0u8..=100 {
        let epsilon_seed = seed ^ (epsilon as u64).wrapping_mul(0x9e3779b97f4a7c15);
        let optimal_wins = parallel_games(
            lut,
            &Game::initial(lut.rules),
            games_per_epsilon,
            epsilon_seed,
            epsilon,
            true,
        );
        let percent = 100.0 * optimal_wins as f64 / games_per_epsilon as f64;
        writeln!(output, "{:.2},{games_per_epsilon},{optimal_wins},{percent:.12}", epsilon as f64 / 100.0).unwrap();
        eprintln!("epsilon={epsilon}% optimal_win={percent:.4}% elapsed={:.1}s", started.elapsed().as_secs_f64());
    }
}

/// Seed used to pick the states compared against simulation. Fixed, so that
/// every shard simulates the same set of states and their counts can be summed.
const COMPARE_STATE_SEED: u64 = 0x1234_5678_9abc_def0;

/// One shard of the Monte Carlo simulations, for running as a Slurm array and
/// combining afterwards.
///
/// Both outputs are binomial counts, so shards combine exactly: summing `games`
/// and the win counts per state (or per epsilon) is identical to having run one
/// long simulation. Only the game RNG depends on `shard_seed`; the states
/// themselves come from COMPARE_STATE_SEED so all shards agree on them.
///
/// Percentages are deliberately not written here -- they are recomputed from the
/// summed counts by scripts/aggregate_simulations.py, because averaging
/// per-shard percentages would be wrong when shards differ in size.
fn write_simulation_shard(
    lut: &Lut,
    output_dir: &Path,
    label: &str,
    compare_states: usize,
    games_per_state: usize,
    games_per_epsilon: usize,
    shard_seed: u64,
) {
    fs::create_dir_all(output_dir).unwrap();
    let name = lut.rules.name();
    let started = Instant::now();

    if compare_states > 0 && games_per_state > 0 {
        let path = output_dir.join(format!("{name}_compare_{label}.csv"));
        let mut output = BufWriter::new(File::create(&path).unwrap());
        writeln!(output, "state,key,predicted_pct,games,light_wins").unwrap();
        let mut sampling_rng = SplitMix64::new(COMPARE_STATE_SEED);
        for state_index in 0..compare_states {
            let (key, predicted) = loop {
                let candidate = lut.key_value_at_global(sampling_rng.index(lut.total));
                if !lut.encoding.decode(candidate.0).finished {
                    break candidate;
                }
            };
            let start_game = lut.encoding.decode(key);
            let game_seed = shard_seed
                ^ key
                ^ (state_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let wins = parallel_games(lut, &start_game, games_per_state, game_seed, 0, false);
            writeln!(output, "{state_index},{key:#x},{predicted:.12},{games_per_state},{wins}").unwrap();
        }
        output.flush().unwrap();
        eprintln!(
            "shard {label}: compare states={compare_states} games_per_state={games_per_state} \
             games={} elapsed={:.1}s",
            compare_states * games_per_state,
            started.elapsed().as_secs_f64()
        );
    }

    if games_per_epsilon > 0 {
        let path = output_dir.join(format!("{name}_epsilon_{label}.csv"));
        let mut output = BufWriter::new(File::create(&path).unwrap());
        writeln!(output, "epsilon,games,optimal_wins").unwrap();
        for epsilon in 0u8..=100 {
            let epsilon_seed = shard_seed
                ^ 0xfedc_ba98_7654_3210
                ^ (epsilon as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let wins = parallel_games(
                lut,
                &Game::initial(lut.rules),
                games_per_epsilon,
                epsilon_seed,
                epsilon,
                true,
            );
            writeln!(output, "{:.2},{games_per_epsilon},{wins}", epsilon as f64 / 100.0).unwrap();
        }
        output.flush().unwrap();
        eprintln!(
            "shard {label}: epsilon levels=101 games_per_epsilon={games_per_epsilon} \
             games={} elapsed={:.1}s",
            101 * games_per_epsilon,
            started.elapsed().as_secs_f64()
        );
    }
}

#[derive(Clone, Copy, Debug)]
struct TrainingMap {
    count: usize,
    base: usize,
}

struct TrainingLut {
    keys: Vec<u32>,
    values: Vec<f64>,
    maps: Vec<TrainingMap>,
    prefix_starts: Vec<Vec<u32>>,
    rules: RuleSet,
    encoding: Encoding,
    metadata: String,
}

impl TrainingLut {
    fn read_percent16(path: &Path) -> Self {
        let started = Instant::now();
        let mut input = BufReader::with_capacity(16 * 1024 * 1024, File::open(path).expect("failed to open LUT"));
        let mut magic = [0u8; 4];
        input.read_exact(&mut magic).unwrap();
        assert_eq!(&magic, b"RGU\0");

        let metadata_len = read_be_u32_from(&mut input) as usize;
        let mut metadata_bytes = vec![0u8; metadata_len];
        input.read_exact(&mut metadata_bytes).unwrap();
        let metadata = String::from_utf8(metadata_bytes).unwrap();
        let metadata_lower = metadata.to_lowercase();
        assert!(
            !metadata_lower.contains("\"value_type\"")
                || metadata_lower.contains("\"value_type\":\"percent16\""),
            "training input must use percent16 values"
        );
        let rules = detect_ruleset(&metadata);

        let map_count = read_be_u32_from(&mut input) as usize;
        let mut counts = Vec::with_capacity(map_count);
        for _ in 0..map_count {
            counts.push(read_be_u32_from(&mut input) as usize);
        }
        let total = counts.iter().sum::<usize>();
        assert!(total <= u32::MAX as usize);

        let mut maps = Vec::with_capacity(map_count);
        let mut base = 0usize;
        for &count in &counts {
            maps.push(TrainingMap { count, base });
            base += count;
        }

        let mut keys = Vec::with_capacity(total);
        read_be_u32_vec(&mut input, total, &mut keys);
        let mut values = Vec::with_capacity(total);
        read_percent16_vec(&mut input, total, &mut values);
        assert_eq!(keys.len(), total);
        assert_eq!(values.len(), total);

        let mut prefix_starts = Vec::with_capacity(map_count);
        for map in &maps {
            let map_keys = &keys[map.base..map.base + map.count];
            let mut starts = vec![0u32; LOOKUP_PREFIX_COUNT + 1];
            let mut index = 0usize;
            for prefix in 0..=LOOKUP_PREFIX_COUNT {
                while index < map.count && ((map_keys[index] as usize) >> LOOKUP_LOWER_BITS) < prefix {
                    index += 1;
                }
                starts[prefix] = index as u32;
            }
            prefix_starts.push(starts);
        }

        eprintln!(
            "loaded training map {}: {} entries, keys={:.3} GB, f64_values={:.3} GB, {:.2}s",
            rules.name(),
            total,
            keys.len() as f64 * 4.0 / 1e9,
            values.len() as f64 * 8.0 / 1e9,
            started.elapsed().as_secs_f64()
        );
        Self {
            keys,
            values,
            maps,
            prefix_starts,
            rules,
            encoding: Encoding::new(rules),
            metadata,
        }
    }

    #[inline]
    fn key_at_global(&self, global: usize) -> u64 {
        for (upper, map) in self.maps.iter().enumerate() {
            if global >= map.base && global < map.base + map.count {
                return ((upper as u64) << 32) | self.keys[global] as u64;
            }
        }
        panic!("global key index out of bounds: {global}");
    }

    #[inline]
    fn lookup_index(&self, key: u64) -> usize {
        let upper = (key >> 32) as usize;
        assert!(upper < self.maps.len(), "upper key missing: {upper}");
        let lower = key as u32;
        let map = self.maps[upper];
        let prefix = (lower >> LOOKUP_LOWER_BITS) as usize;
        let map_keys = &self.keys[map.base..map.base + map.count];
        let starts = &self.prefix_starts[upper];
        let lo = starts[prefix] as usize;
        let hi = starts[prefix + 1] as usize;
        match map_keys[lo..hi].binary_search(&lower) {
            Ok(local) => map.base + lo + local,
            Err(_) => panic!("key missing from training LUT: {key:#x}"),
        }
    }

    #[inline]
    fn light_win_percent(&self, game: &Game, values: &[f64]) -> f64 {
        if game.finished {
            return if game.light_score >= self.rules.pieces() { 100.0 } else { 0.0 };
        }
        let index = self.lookup_index(self.encoding.encode_symmetrical(game));
        let value = values[index];
        if game.is_light_turn { value } else { 100.0 - value }
    }

    fn bellman_key(&self, key: u64, values: &[f64]) -> f64 {
        let game = self.encoding.decode(key);
        debug_assert!(!game.finished);
        let mut total = 0.0;
        let mut moves = [0i8; 8];
        for (roll, &probability) in self.rules.roll_probabilities().iter().enumerate() {
            if probability == 0.0 {
                continue;
            }
            let mut rolled = game.clone();
            let move_count = rolled.apply_roll(roll as u8, &mut moves);
            let best = if move_count == 0 {
                self.light_win_percent(&rolled, values)
            } else {
                let mut best = f64::NEG_INFINITY;
                for &source in &moves[..move_count] {
                    let mut next = rolled.clone();
                    next.apply_move(source, self.rules);
                    best = best.max(self.light_win_percent(&next, values));
                }
                best
            };
            total += probability * best;
        }
        total
    }

    fn write_f64(&self, output: &Path, target_precision: f64, training_precision: f64) {
        let partial = output.with_extension("rgu.partial");
        let mut writer = BufWriter::with_capacity(
            16 * 1024 * 1024,
            File::create(&partial).expect("failed to create f64 output"),
        );
        let metadata = format!(
            "{{\"value_type\":\"f64\",\"target-precision\":{target_precision:.17e},\"training-precision\":{training_precision:.17e},{}",
            &self.metadata[1..]
        );
        writer.write_all(b"RGU\0").unwrap();
        writer.write_all(&(metadata.len() as u32).to_be_bytes()).unwrap();
        writer.write_all(metadata.as_bytes()).unwrap();
        writer.write_all(&(self.maps.len() as u32).to_be_bytes()).unwrap();
        for map in &self.maps {
            writer.write_all(&(map.count as u32).to_be_bytes()).unwrap();
        }
        let mut key_buffer = Vec::with_capacity(4 * 1_048_576);
        for chunk in self.keys.chunks(1_048_576) {
            key_buffer.clear();
            for &key in chunk {
                key_buffer.extend_from_slice(&key.to_be_bytes());
            }
            writer.write_all(&key_buffer).unwrap();
        }
        let mut value_buffer = Vec::with_capacity(8 * 524_288);
        for chunk in self.values.chunks(524_288) {
            value_buffer.clear();
            for &value in chunk {
                value_buffer.extend_from_slice(&value.to_be_bytes());
            }
            writer.write_all(&value_buffer).unwrap();
        }
        writer.flush().unwrap();
        drop(writer);
        fs::rename(&partial, output).expect("failed to move completed f64 output into place");
    }
}

fn read_be_u32_from<R: Read>(input: &mut R) -> u32 {
    let mut bytes = [0u8; 4];
    input.read_exact(&mut bytes).unwrap();
    u32::from_be_bytes(bytes)
}

fn read_be_u32_vec<R: Read>(input: &mut R, count: usize, output: &mut Vec<u32>) {
    let mut bytes = vec![0u8; 4 * 1_048_576];
    let mut remaining = count;
    while remaining > 0 {
        let chunk = remaining.min(1_048_576);
        input.read_exact(&mut bytes[..4 * chunk]).unwrap();
        output.extend(
            bytes[..4 * chunk]
                .chunks_exact(4)
                .map(|value| u32::from_be_bytes(value.try_into().unwrap())),
        );
        remaining -= chunk;
    }
}

fn read_percent16_vec<R: Read>(input: &mut R, count: usize, output: &mut Vec<f64>) {
    let mut bytes = vec![0u8; 2 * 1_048_576];
    let mut remaining = count;
    while remaining > 0 {
        let chunk = remaining.min(1_048_576);
        input.read_exact(&mut bytes[..2 * chunk]).unwrap();
        output.extend(bytes[..2 * chunk].chunks_exact(2).map(|value| {
            u16::from_be_bytes(value.try_into().unwrap()) as f64 * 100.0 / 65535.0
        }));
        remaining -= chunk;
    }
}

fn score_pairs(piece_count: u8) -> Vec<(u8, u8)> {
    let mut pairs = Vec::new();
    for min_score in (0..piece_count).rev() {
        for max_score in (min_score..piece_count).rev() {
            pairs.push((min_score, max_score));
        }
    }
    pairs
}

fn layer_file(layer_dir: &Path, pair: (u8, u8)) -> PathBuf {
    layer_dir.join(format!("layer_{:02}_{:02}.bin", pair.0, pair.1))
}

fn build_layer_files(lut: &TrainingLut, layer_dir: &Path) -> Vec<(u8, u8)> {
    fs::create_dir_all(layer_dir).unwrap();
    let pairs = score_pairs(lut.rules.pieces());
    let mut pair_to_index = vec![usize::MAX; 64];
    for (index, &(min_score, max_score)) in pairs.iter().enumerate() {
        pair_to_index[min_score as usize * 8 + max_score as usize] = index;
    }
    let mut writers = pairs
        .iter()
        .map(|&pair| BufWriter::with_capacity(1024 * 1024, File::create(layer_file(layer_dir, pair)).unwrap()))
        .collect::<Vec<_>>();
    let started = Instant::now();
    let mut nonterminal = 0usize;
    for global in 0..lut.keys.len() {
        let key = lut.key_at_global(global);
        let (light_score, dark_score) = lut.encoding.scores(key);
        if light_score >= lut.rules.pieces() {
            continue;
        }
        let min_score = light_score.min(dark_score);
        let max_score = light_score.max(dark_score);
        let layer = pair_to_index[min_score as usize * 8 + max_score as usize];
        assert_ne!(layer, usize::MAX);
        writers[layer].write_all(&(global as u32).to_le_bytes()).unwrap();
        nonterminal += 1;
        if global > 0 && global % 50_000_000 == 0 {
            eprintln!(
                "layer_index states={}/{} nonterminal={} elapsed_seconds={:.1}",
                global,
                lut.keys.len(),
                nonterminal,
                started.elapsed().as_secs_f64()
            );
        }
    }
    for writer in &mut writers {
        writer.flush().unwrap();
    }
    eprintln!(
        "layer_index complete states={} nonterminal={} elapsed_seconds={:.1}",
        lut.keys.len(),
        nonterminal,
        started.elapsed().as_secs_f64()
    );
    pairs
}

fn read_layer_indices(path: &Path) -> Vec<u32> {
    let size = fs::metadata(path).unwrap().len() as usize;
    assert_eq!(size % 4, 0);
    let count = size / 4;
    let mut input = BufReader::with_capacity(16 * 1024 * 1024, File::open(path).unwrap());
    let mut indices = Vec::with_capacity(count);
    let mut bytes = vec![0u8; 4 * 1_048_576];
    let mut remaining = count;
    while remaining > 0 {
        let chunk = remaining.min(1_048_576);
        input.read_exact(&mut bytes[..4 * chunk]).unwrap();
        indices.extend(
            bytes[..4 * chunk]
                .chunks_exact(4)
                .map(|value| u32::from_le_bytes(value.try_into().unwrap())),
        );
        remaining -= chunk;
    }
    indices
}

fn training_iteration(lut: &TrainingLut, indices: &[u32]) -> (Vec<f64>, f64) {
    let threads = thread::available_parallelism().map(usize::from).unwrap_or(1).max(1);
    let chunk_size = (indices.len() + threads - 1) / threads;
    let mut updates = vec![0.0f64; indices.len()];
    let values = &lut.values;
    let max_delta = thread::scope(|scope| {
        let mut handles = Vec::new();
        for (index_chunk, update_chunk) in indices.chunks(chunk_size).zip(updates.chunks_mut(chunk_size)) {
            handles.push(scope.spawn(move || {
                let mut delta = 0.0f64;
                for (slot, &global) in update_chunk.iter_mut().zip(index_chunk) {
                    let global = global as usize;
                    let value = lut.bellman_key(lut.key_at_global(global), values);
                    delta = delta.max((value - values[global]).abs());
                    *slot = value;
                }
                delta
            }));
        }
        handles.into_iter().map(|handle| handle.join().unwrap()).fold(0.0, f64::max)
    });
    (updates, max_delta)
}

const CHECKPOINT_MAGIC: &[u8; 8] = b"RGUCHK1\0";

fn load_checkpoint(
    checkpoint: &Path,
    lut: &mut TrainingLut,
    layer_dir: &Path,
    pairs: &[(u8, u8)],
    required_tolerance: f64,
) -> (Vec<bool>, Vec<f64>) {
    let mut completed = vec![false; pairs.len()];
    let mut precisions = vec![0.0; pairs.len()];
    if !checkpoint.exists() {
        let mut output = BufWriter::new(File::create(checkpoint).unwrap());
        output.write_all(CHECKPOINT_MAGIC).unwrap();
        output.write_all(&(lut.keys.len() as u64).to_le_bytes()).unwrap();
        output.flush().unwrap();
        return (completed, precisions);
    }

    let file_len = fs::metadata(checkpoint).unwrap().len();
    let mut input = BufReader::with_capacity(16 * 1024 * 1024, File::open(checkpoint).unwrap());
    let mut magic = [0u8; 8];
    input.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, CHECKPOINT_MAGIC);
    let mut total_bytes = [0u8; 8];
    input.read_exact(&mut total_bytes).unwrap();
    assert_eq!(u64::from_le_bytes(total_bytes) as usize, lut.keys.len());
    let mut offset = 16u64;

    loop {
        if file_len < offset + 18 {
            break;
        }
        let mut header = [0u8; 18];
        input.read_exact(&mut header).unwrap();
        let pair = (header[0], header[1]);
        let count = u64::from_le_bytes(header[2..10].try_into().unwrap()) as usize;
        let precision = f64::from_le_bytes(header[10..18].try_into().unwrap());
        let record_end = offset + 18 + 8 * count as u64;
        if file_len < record_end {
            break;
        }
        let layer_index = pairs.iter().position(|&candidate| candidate == pair).unwrap();
        let indices = read_layer_indices(&layer_file(layer_dir, pair));
        assert_eq!(indices.len(), count);
        let mut value_bytes = [0u8; 8];
        for &global in &indices {
            input.read_exact(&mut value_bytes).unwrap();
            lut.values[global as usize] = f64::from_le_bytes(value_bytes);
        }
        completed[layer_index] = precision <= required_tolerance;
        precisions[layer_index] = precision;
        offset = record_end;
        eprintln!("checkpoint restored scores=[{},{}] states={} precision={:.3e}", pair.0, pair.1, count, precision);
    }
    OpenOptions::new().write(true).open(checkpoint).unwrap().set_len(offset).unwrap();
    (completed, precisions)
}

fn append_checkpoint(checkpoint: &Path, pair: (u8, u8), precision: f64, indices: &[u32], values: &[f64]) {
    let file = OpenOptions::new().append(true).open(checkpoint).unwrap();
    let mut output = BufWriter::with_capacity(16 * 1024 * 1024, file);
    output.write_all(&[pair.0, pair.1]).unwrap();
    output.write_all(&(indices.len() as u64).to_le_bytes()).unwrap();
    output.write_all(&precision.to_le_bytes()).unwrap();
    let mut buffer = Vec::with_capacity(8 * 524_288);
    for chunk in indices.chunks(524_288) {
        buffer.clear();
        for &global in chunk {
            buffer.extend_from_slice(&values[global as usize].to_le_bytes());
        }
        output.write_all(&buffer).unwrap();
    }
    output.flush().unwrap();
    output.get_ref().sync_data().unwrap();
}

/// Which iteration scheme solves each score layer.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Strategy {
    /// Successors regenerated every sweep; deterministic Jacobi updates.
    OnDemandJacobi,
    /// Successor indices materialised once per layer; in-place Gauss-Seidel
    /// sweeps, with convergence certified by a deterministic residual pass.
    PrecomputedGaussSeidel,
}

impl Strategy {
    fn name(self) -> &'static str {
        match self {
            Self::OnDemandJacobi => "ondemand-jacobi",
            Self::PrecomputedGaussSeidel => "precomputed-gauss-seidel",
        }
    }
}

/// Solve one score layer with in-place Gauss-Seidel sweeps over a precomputed
/// successor table, returning the certified residual.
///
/// Gauss-Seidel sweep deltas are timing dependent, so they are used only as a
/// cheap progress signal: once a sweep looks converged, a deterministic
/// residual pass decides whether the layer is actually done. The committed
/// values therefore satisfy exactly the same criterion as the Jacobi path.
fn solve_layer_gauss_seidel(
    lut: &mut TrainingLut,
    indices: &[u32],
    pair: (u8, u8),
    tolerance: f64,
    max_iterations: usize,
    started: &Instant,
) -> f64 {
    let build_started = Instant::now();
    let successors = build_layer_successors(lut, indices);
    eprintln!(
        "train scores=[{},{}] successor_table entries={} bytes={:.3}GB build_seconds={:.2}",
        pair.0,
        pair.1,
        successors.entries.len(),
        successors.bytes() as f64 / 1e9,
        build_started.elapsed().as_secs_f64()
    );

    // Guard against a successor-table bug silently producing wrong values: the
    // precomputed Bellman update must agree exactly with the on-demand one.
    // build_layer_successors already proves every successor key is present in
    // the map (lookup_index panics otherwise); this checks the arithmetic.
    let mut worst = 0.0f64;
    let mut rng = SplitMix64::new(0x5bd1_e995_1234_9f3b ^ ((pair.0 as u64) << 8) ^ pair.1 as u64);
    let checks = indices.len().min(2_000);
    for _ in 0..checks {
        let position = rng.index(indices.len());
        let global = indices[position] as usize;
        let expected = lut.bellman_key(lut.key_at_global(global), &lut.values);
        let actual = successors.bellman(position, &lut.values);
        worst = worst.max((expected - actual).abs());
    }
    assert!(
        worst == 0.0,
        "layer {pair:?}: precomputed successors disagree with on-demand Bellman by {worst:.3e}"
    );
    eprintln!("train scores=[{},{}] successor_check_states={checks} max_abs_diff=0", pair.0, pair.1);

    let mut residual = f64::INFINITY;
    let mut sweeps = 0usize;
    while sweeps < max_iterations {
        let sweep_started = Instant::now();
        let delta = gauss_seidel_precomputed(&successors, indices, &mut lut.values);
        sweeps += 1;
        if sweeps <= 5 || sweeps % 10 == 0 || delta <= tolerance {
            eprintln!(
                "train scores=[{},{}] sweep={} max_delta={:.12e} seconds={:.2}",
                pair.0,
                pair.1,
                sweeps,
                delta,
                sweep_started.elapsed().as_secs_f64()
            );
        }
        if delta <= tolerance {
            residual = residual_precomputed(&successors, indices, &lut.values);
            eprintln!(
                "train scores=[{},{}] certified_residual={:.12e} after sweeps={} elapsed_seconds={:.1}",
                pair.0,
                pair.1,
                residual,
                sweeps,
                started.elapsed().as_secs_f64()
            );
            if residual <= tolerance {
                break;
            }
        }
    }
    if residual > tolerance {
        residual = residual_precomputed(&successors, indices, &lut.values);
    }
    residual
}

/// Where the iteration starts from.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Init {
    /// The published Percent16 values, which are the solution already, quantised.
    /// Refining them to f64 is the cheap case: the starting residual is ~1e-3.
    Published,
    /// A flat 50% for every state, ignoring the published values and keeping only
    /// the map's state keys. This is a solve from scratch in the sense that no
    /// prior solution is used, and it is the honest number to quote for "how long
    /// does it take to solve the game", as opposed to "to refine a known
    /// solution". Terminal states are never read through the value array (their
    /// value comes from the game result), so overwriting every entry is safe.
    Naive,
}

impl Init {
    fn name(self) -> &'static str {
        match self {
            Self::Published => "published",
            Self::Naive => "naive",
        }
    }
}

fn train_f64(
    input: &Path,
    output: &Path,
    tolerance: f64,
    max_iterations: usize,
    strategy: Strategy,
    init: Init,
) {
    assert!(tolerance > 0.0);
    eprintln!("strategy={} init={}", strategy.name(), init.name());
    let mut lut = TrainingLut::read_percent16(input);
    if init == Init::Naive {
        // Discard the published values; keep only the state keys.
        lut.values.iter_mut().for_each(|value| *value = 50.0);
        eprintln!("init: discarded published values, all states set to 50%");
    }
    let layer_dir = output.with_extension("layers");
    let pairs = build_layer_files(&lut, &layer_dir);
    let checkpoint = output.with_extension("checkpoint");
    let (completed, mut precisions) = load_checkpoint(&checkpoint, &mut lut, &layer_dir, &pairs, tolerance);
    let started = Instant::now();

    for (layer_index, &pair) in pairs.iter().enumerate() {
        if completed[layer_index] {
            continue;
        }
        let indices = read_layer_indices(&layer_file(&layer_dir, pair));
        eprintln!(
            "train scores=[{},{}] states={} start elapsed_seconds={:.1}",
            pair.0,
            pair.1,
            indices.len(),
            started.elapsed().as_secs_f64()
        );
        let final_delta = match strategy {
            Strategy::OnDemandJacobi => {
                let mut final_delta = f64::INFINITY;
                for iteration in 1..=max_iterations {
                    let iteration_started = Instant::now();
                    let (updates, delta) = training_iteration(&lut, &indices);
                    for (&global, value) in indices.iter().zip(updates) {
                        lut.values[global as usize] = value;
                    }
                    final_delta = delta;
                    if iteration <= 5 || iteration % 10 == 0 || delta <= tolerance {
                        eprintln!(
                            "train scores=[{},{}] iteration={} max_delta={:.12e} seconds={:.2}",
                            pair.0,
                            pair.1,
                            iteration,
                            delta,
                            iteration_started.elapsed().as_secs_f64()
                        );
                    }
                    if delta <= tolerance {
                        break;
                    }
                }
                final_delta
            }
            Strategy::PrecomputedGaussSeidel => solve_layer_gauss_seidel(
                &mut lut,
                &indices,
                pair,
                tolerance,
                max_iterations,
                &started,
            ),
        };
        assert!(final_delta <= tolerance, "layer {pair:?} did not converge within {max_iterations} iterations");
        precisions[layer_index] = final_delta;
        append_checkpoint(&checkpoint, pair, final_delta, &indices, &lut.values);
        eprintln!(
            "train scores=[{},{}] complete precision={:.12e} elapsed_seconds={:.1}",
            pair.0,
            pair.1,
            final_delta,
            started.elapsed().as_secs_f64()
        );
    }

    let training_precision = precisions.into_iter().fold(0.0f64, f64::max);
    eprintln!("writing f64 map {} precision={:.12e}", output.display(), training_precision);
    lut.write_f64(output, tolerance, training_precision);
}

fn preflight_training(input: &Path, samples: usize) {
    let lut = TrainingLut::read_percent16(input);
    let mut rng = SplitMix64::new(0x763b_91a4_ef02_cd58);
    let mut checked = 0usize;
    let mut max_residual = 0.0f64;
    while checked < samples {
        let global = rng.index(lut.keys.len());
        let key = lut.key_at_global(global);
        let game = lut.encoding.decode(key);
        let scores = lut.encoding.scores(key);
        assert_eq!(scores, (game.light_score, game.dark_score));
        assert_eq!(lut.lookup_index(key), global);
        if game.finished {
            continue;
        }
        let updated = lut.bellman_key(key, &lut.values);
        max_residual = max_residual.max((updated - lut.values[global]).abs());
        checked += 1;
    }
    println!("ruleset={}", lut.rules.name());
    println!("entries={}", lut.keys.len());
    println!("samples={checked}");
    println!("sample_max_bellman_residual={max_residual:.12e}");
}

// ---------------------------------------------------------------------------
// Precomputed-successor solver (benchmark path)
//
// The on-demand path re-decodes each state and re-runs a binary-search lookup
// for every successor on every iteration. Because a score layer is iterated
// many times, it is cheaper to materialise the successor indices for the layer
// once and then let each sweep be a pure gather over that table.
//
// Only the layer currently being solved needs a successor table: successors
// outside the layer have frozen values, so they are read by index like any
// other. That keeps the table proportional to the layer rather than to the
// whole 500,981,472-state map.
// ---------------------------------------------------------------------------

/// Successor entry: a sentinel for a terminal outcome, or a state index with a
/// flag saying the stored light-win value must be complemented (dark to move).
const SUCCESSOR_LIGHT_WIN: u32 = u32::MAX;
const SUCCESSOR_DARK_WIN: u32 = u32::MAX - 1;
const SUCCESSOR_COMPLEMENT: u32 = 1 << 31;
const SUCCESSOR_INDEX_MASK: u32 = SUCCESSOR_COMPLEMENT - 1;

struct LayerSuccessors {
    /// Rolls with nonzero probability, paired with that probability.
    active_rolls: Vec<(u8, f64)>,
    /// CSR offsets, one run per (layer position, active roll).
    offsets: Vec<u64>,
    entries: Vec<u32>,
}

impl LayerSuccessors {
    #[inline]
    fn resolve(entry: u32, values: &[f64]) -> f64 {
        match entry {
            SUCCESSOR_LIGHT_WIN => 100.0,
            SUCCESSOR_DARK_WIN => 0.0,
            _ => {
                let value = values[(entry & SUCCESSOR_INDEX_MASK) as usize];
                if entry & SUCCESSOR_COMPLEMENT != 0 { 100.0 - value } else { value }
            }
        }
    }

    #[inline]
    fn bellman(&self, position: usize, values: &[f64]) -> f64 {
        let rolls = self.active_rolls.len();
        let mut total = 0.0;
        for (roll_slot, &(_, probability)) in self.active_rolls.iter().enumerate() {
            let start = self.offsets[position * rolls + roll_slot] as usize;
            let end = self.offsets[position * rolls + roll_slot + 1] as usize;
            let mut best = f64::NEG_INFINITY;
            for &entry in &self.entries[start..end] {
                let candidate = Self::resolve(entry, values);
                if candidate > best {
                    best = candidate;
                }
            }
            total += probability * best;
        }
        total
    }

    fn bytes(&self) -> usize {
        self.offsets.len() * 8 + self.entries.len() * 4
    }
}

/// Encode one successor game state as a `LayerSuccessors` entry.
fn successor_entry(lut: &TrainingLut, game: &Game) -> u32 {
    if game.finished {
        return if game.light_score >= lut.rules.pieces() {
            SUCCESSOR_LIGHT_WIN
        } else {
            SUCCESSOR_DARK_WIN
        };
    }
    let index = lut.lookup_index(lut.encoding.encode_symmetrical(game));
    assert!(index as u32 <= SUCCESSOR_INDEX_MASK, "state index does not fit in 31 bits: {index}");
    let mut entry = index as u32;
    if !game.is_light_turn {
        entry |= SUCCESSOR_COMPLEMENT;
    }
    entry
}

/// Append the successor entries for one state and one roll, returning the count.
fn push_successors(lut: &TrainingLut, key: u64, roll: u8, entries: &mut Vec<u32>) -> usize {
    let game = lut.encoding.decode(key);
    let mut moves = [0i8; 8];
    let mut rolled = game;
    let move_count = rolled.apply_roll(roll, &mut moves);
    if move_count == 0 {
        entries.push(successor_entry(lut, &rolled));
        return 1;
    }
    for &source in &moves[..move_count] {
        let mut next = rolled.clone();
        next.apply_move(source, lut.rules);
        entries.push(successor_entry(lut, &next));
    }
    move_count
}

fn build_layer_successors(lut: &TrainingLut, indices: &[u32]) -> LayerSuccessors {
    let active_rolls = lut
        .rules
        .roll_probabilities()
        .iter()
        .enumerate()
        .filter(|(_, &probability)| probability > 0.0)
        .map(|(roll, &probability)| (roll as u8, probability))
        .collect::<Vec<_>>();
    let rolls = active_rolls.len();

    let threads = thread::available_parallelism().map(usize::from).unwrap_or(1).max(1);
    let chunk_size = (indices.len() + threads - 1) / threads;

    // Each thread builds a contiguous chunk, so concatenating the per-chunk
    // buffers in order reproduces the global CSR ordering.
    let chunks: Vec<(Vec<u32>, Vec<u32>)> = thread::scope(|scope| {
        let mut handles = Vec::new();
        for index_chunk in indices.chunks(chunk_size.max(1)) {
            let active_rolls = &active_rolls;
            handles.push(scope.spawn(move || {
                let mut entries = Vec::with_capacity(index_chunk.len() * rolls * 3);
                let mut counts = Vec::with_capacity(index_chunk.len() * rolls);
                for &global in index_chunk {
                    let key = lut.key_at_global(global as usize);
                    for &(roll, _) in active_rolls.iter() {
                        let count = push_successors(lut, key, roll, &mut entries);
                        counts.push(count as u32);
                    }
                }
                (counts, entries)
            }));
        }
        handles.into_iter().map(|handle| handle.join().unwrap()).collect()
    });

    let total_entries = chunks.iter().map(|(_, entries)| entries.len()).sum::<usize>();
    let mut offsets = Vec::with_capacity(indices.len() * rolls + 1);
    let mut entries = Vec::with_capacity(total_entries);
    let mut running = 0u64;
    offsets.push(0);
    for (counts, chunk_entries) in &chunks {
        for &count in counts {
            running += count as u64;
            offsets.push(running);
        }
        entries.extend_from_slice(chunk_entries);
    }
    assert_eq!(offsets.len(), indices.len() * rolls + 1);
    assert_eq!(entries.len(), total_entries);

    LayerSuccessors { active_rolls, offsets, entries }
}

/// Deterministic Jacobi sweep over a precomputed layer table.
fn jacobi_precomputed(successors: &LayerSuccessors, indices: &[u32], values: &[f64]) -> (Vec<f64>, f64) {
    let threads = thread::available_parallelism().map(usize::from).unwrap_or(1).max(1);
    let chunk_size = (indices.len() + threads - 1) / threads;
    let mut updates = vec![0.0f64; indices.len()];
    let max_delta = thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut base = 0usize;
        for (index_chunk, update_chunk) in
            indices.chunks(chunk_size.max(1)).zip(updates.chunks_mut(chunk_size.max(1)))
        {
            let start = base;
            base += index_chunk.len();
            handles.push(scope.spawn(move || {
                let mut delta = 0.0f64;
                for (offset, (slot, &global)) in
                    update_chunk.iter_mut().zip(index_chunk).enumerate()
                {
                    let value = successors.bellman(start + offset, values);
                    delta = delta.max((value - values[global as usize]).abs());
                    *slot = value;
                }
                delta
            }));
        }
        handles.into_iter().map(|handle| handle.join().unwrap()).fold(0.0, f64::max)
    });
    (updates, max_delta)
}

/// Max residual |T(v) - v| over a layer, without writing values or allocating
/// an update buffer. This is the deterministic convergence certificate.
fn residual_precomputed(successors: &LayerSuccessors, indices: &[u32], values: &[f64]) -> f64 {
    let threads = thread::available_parallelism().map(usize::from).unwrap_or(1).max(1);
    let chunk_size = (indices.len() + threads - 1) / threads;
    thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut base = 0usize;
        for index_chunk in indices.chunks(chunk_size.max(1)) {
            let start = base;
            base += index_chunk.len();
            handles.push(scope.spawn(move || {
                let mut delta = 0.0f64;
                for (offset, &global) in index_chunk.iter().enumerate() {
                    let value = successors.bellman(start + offset, values);
                    delta = delta.max((value - values[global as usize]).abs());
                }
                delta
            }));
        }
        handles.into_iter().map(|handle| handle.join().unwrap()).fold(0.0, f64::max)
    })
}

/// Parallel in-place (Gauss-Seidel style) sweep over a precomputed layer table.
///
/// Threads publish updated values as they go, so later reads within the same
/// sweep see newer values -- the asynchronous-value-iteration analogue of the
/// serial Gauss-Seidel sweep in the Julia solver. Which updates a thread
/// observes is timing dependent, so the returned delta is only an indication of
/// progress; convergence is certified by a separate deterministic Jacobi
/// residual sweep.
///
/// Values are accessed as relaxed atomics purely to make the concurrent
/// reads/writes well defined; on x86-64 these compile to plain loads/stores.
fn gauss_seidel_precomputed(successors: &LayerSuccessors, indices: &[u32], values: &mut [f64]) -> f64 {
    let threads = thread::available_parallelism().map(usize::from).unwrap_or(1).max(1);
    let chunk_size = (indices.len() + threads - 1) / threads;
    let length = values.len();
    let pointer = values.as_mut_ptr();
    // Safety: f64 and AtomicU64 share size and alignment, and every access to
    // the slice below goes through relaxed atomic operations for the duration
    // of this function, so concurrent access is defined and cannot tear.
    let atomics: &[AtomicU64] = unsafe { std::slice::from_raw_parts(pointer as *const AtomicU64, length) };

    thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut base = 0usize;
        for index_chunk in indices.chunks(chunk_size.max(1)) {
            let start = base;
            base += index_chunk.len();
            handles.push(scope.spawn(move || {
                let mut delta = 0.0f64;
                for (offset, &global) in index_chunk.iter().enumerate() {
                    let position = start + offset;
                    let value = bellman_atomic(successors, position, atomics);
                    let previous =
                        f64::from_bits(atomics[global as usize].load(AtomicOrdering::Relaxed));
                    delta = delta.max((value - previous).abs());
                    atomics[global as usize].store(value.to_bits(), AtomicOrdering::Relaxed);
                }
                delta
            }));
        }
        handles.into_iter().map(|handle| handle.join().unwrap()).fold(0.0, f64::max)
    })
}

#[inline]
fn bellman_atomic(successors: &LayerSuccessors, position: usize, values: &[AtomicU64]) -> f64 {
    let rolls = successors.active_rolls.len();
    let mut total = 0.0;
    for (roll_slot, &(_, probability)) in successors.active_rolls.iter().enumerate() {
        let start = successors.offsets[position * rolls + roll_slot] as usize;
        let end = successors.offsets[position * rolls + roll_slot + 1] as usize;
        let mut best = f64::NEG_INFINITY;
        for &entry in &successors.entries[start..end] {
            let candidate = match entry {
                SUCCESSOR_LIGHT_WIN => 100.0,
                SUCCESSOR_DARK_WIN => 0.0,
                _ => {
                    let value = f64::from_bits(
                        values[(entry & SUCCESSOR_INDEX_MASK) as usize].load(AtomicOrdering::Relaxed),
                    );
                    if entry & SUCCESSOR_COMPLEMENT != 0 { 100.0 - value } else { value }
                }
            };
            if candidate > best {
                best = candidate;
            }
        }
        total += probability * best;
    }
    total
}

/// Compare the converged layer values recorded in a checkpoint against a
/// finished f64 map, per score layer. Used to cross-check two independent runs
/// (different machines, different iteration schemes) against each other.
///
/// Read-only: unlike `load_checkpoint` this never truncates a trailing partial
/// record, it just stops there.
fn compare_checkpoint(checkpoint: &Path, layer_dir: &Path, model: &Path) {
    let lut = Lut::read(model);
    let file_len = fs::metadata(checkpoint).unwrap().len();
    let mut input = BufReader::with_capacity(16 * 1024 * 1024, File::open(checkpoint).unwrap());
    let mut magic = [0u8; 8];
    input.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, CHECKPOINT_MAGIC, "not a solver checkpoint: {}", checkpoint.display());
    let mut total_bytes = [0u8; 8];
    input.read_exact(&mut total_bytes).unwrap();
    let checkpoint_states = u64::from_le_bytes(total_bytes) as usize;
    println!("checkpoint_states={checkpoint_states}");
    println!("model_states={}", lut.total);
    assert_eq!(checkpoint_states, lut.total, "checkpoint and model cover different state counts");

    let mut offset = 16u64;
    let mut layers = 0usize;
    let mut compared = 0usize;
    let mut worst_overall = 0.0f64;
    loop {
        if file_len < offset + 18 {
            break;
        }
        let mut header = [0u8; 18];
        input.read_exact(&mut header).unwrap();
        let pair = (header[0], header[1]);
        let count = u64::from_le_bytes(header[2..10].try_into().unwrap()) as usize;
        let precision = f64::from_le_bytes(header[10..18].try_into().unwrap());
        let record_end = offset + 18 + 8 * count as u64;
        if file_len < record_end {
            println!("stopping at incomplete trailing record for scores=[{},{}]", pair.0, pair.1);
            break;
        }
        let indices = read_layer_indices(&layer_file(layer_dir, pair));
        assert_eq!(indices.len(), count, "layer [{},{}] index count does not match record", pair.0, pair.1);
        let mut worst = 0.0f64;
        let mut value_bytes = [0u8; 8];
        for &global in &indices {
            input.read_exact(&mut value_bytes).unwrap();
            let recorded = f64::from_le_bytes(value_bytes);
            let stored = lut.value_at_global(global as usize);
            worst = worst.max((recorded - stored).abs());
        }
        worst_overall = worst_overall.max(worst);
        compared += count;
        layers += 1;
        println!(
            "layer scores=[{},{}] states={} checkpoint_precision={:.3e} max_abs_diff={:.6e}",
            pair.0, pair.1, count, precision, worst
        );
        offset = record_end;
    }
    println!("layers_compared={layers} states_compared={compared}");
    println!("max_abs_diff_overall={worst_overall:.6e}");
}

fn peak_rss_bytes() -> u64 {
    // VmHWM is the kernel's high-water mark; absent on non-Linux hosts.
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kilobytes = rest.trim().trim_end_matches(" kB").trim();
            return kilobytes.parse::<u64>().unwrap_or(0) * 1024;
        }
    }
    0
}

/// Benchmark one score layer: preprocessing cost, per-sweep wall time and
/// convergence rate for the on-demand Jacobi, precomputed Jacobi and
/// precomputed Gauss-Seidel strategies.
fn bench_layer(input: &Path, layer: (u8, u8), sweeps: usize, work_dir: &Path) {
    let threads = thread::available_parallelism().map(usize::from).unwrap_or(1).max(1);
    println!("threads={threads}");

    let load_started = Instant::now();
    let mut lut = TrainingLut::read_percent16(input);
    println!("map_load_seconds={:.2}", load_started.elapsed().as_secs_f64());
    println!("total_states={}", lut.keys.len());

    // Correctness gate: encoding round trips and successor agreement.
    validate_encoding(&lut, 10_000);

    fs::create_dir_all(work_dir).unwrap();
    let layer_dir = work_dir.join("bench.layers");
    let index_started = Instant::now();
    if !layer_file(&layer_dir, layer).exists() {
        build_layer_files(&lut, &layer_dir);
    }
    println!("layer_index_seconds={:.2}", index_started.elapsed().as_secs_f64());

    // Every layer's state count, so total runtime can be extrapolated from a
    // per-sweep cost measured on one layer.
    let mut layer_total = 0u64;
    for pair in score_pairs(lut.rules.pieces()) {
        let path = layer_file(&layer_dir, pair);
        let states = fs::metadata(&path).map(|meta| meta.len() / 4).unwrap_or(0);
        layer_total += states;
        println!("layer_size scores=[{},{}] states={}", pair.0, pair.1, states);
    }
    println!("layer_states_total={layer_total}");

    let indices = read_layer_indices(&layer_file(&layer_dir, layer));
    println!("layer=[{},{}] layer_states={}", layer.0, layer.1, indices.len());
    if indices.is_empty() {
        println!("layer is empty; nothing to benchmark");
        return;
    }

    // --- Strategy A: on-demand successors, Jacobi (current production path).
    let mut values_a = lut.values.clone();
    let mut deltas_a = Vec::new();
    let a_started = Instant::now();
    for _ in 0..sweeps {
        std::mem::swap(&mut lut.values, &mut values_a);
        let (updates, delta) = training_iteration(&lut, &indices);
        std::mem::swap(&mut lut.values, &mut values_a);
        for (&global, value) in indices.iter().zip(updates) {
            values_a[global as usize] = value;
        }
        deltas_a.push(delta);
    }
    let a_seconds = a_started.elapsed().as_secs_f64();
    println!(
        "ondemand_jacobi sweeps={} total_seconds={:.3} seconds_per_sweep={:.3}",
        sweeps,
        a_seconds,
        a_seconds / sweeps as f64
    );
    println!("ondemand_jacobi_deltas={}", format_deltas(&deltas_a));

    // --- Preprocessing for the precomputed strategies.
    let build_started = Instant::now();
    let successors = build_layer_successors(&lut, &indices);
    let build_seconds = build_started.elapsed().as_secs_f64();
    println!(
        "successor_build_seconds={:.3} successor_entries={} successor_bytes={:.3}GB bytes_per_state={:.1}",
        build_seconds,
        successors.entries.len(),
        successors.bytes() as f64 / 1e9,
        successors.bytes() as f64 / indices.len() as f64
    );

    // Agreement check: precomputed Bellman must match the on-demand Bellman.
    let mut worst = 0.0f64;
    let mut rng = SplitMix64::new(0x5bd1_e995_1234_9f3b);
    for _ in 0..10_000 {
        let position = rng.index(indices.len());
        let global = indices[position] as usize;
        let expected = lut.bellman_key(lut.key_at_global(global), &lut.values);
        let actual = successors.bellman(position, &lut.values);
        worst = worst.max((expected - actual).abs());
    }
    println!("precomputed_vs_ondemand_max_abs_diff={worst:.3e}");
    assert!(worst == 0.0, "precomputed successors disagree with on-demand Bellman");

    // --- Strategy B: precomputed successors, Jacobi.
    let mut values_b = lut.values.clone();
    let mut deltas_b = Vec::new();
    let b_started = Instant::now();
    for _ in 0..sweeps {
        let (updates, delta) = jacobi_precomputed(&successors, &indices, &values_b);
        for (&global, value) in indices.iter().zip(updates) {
            values_b[global as usize] = value;
        }
        deltas_b.push(delta);
    }
    let b_seconds = b_started.elapsed().as_secs_f64();
    println!(
        "precomputed_jacobi sweeps={} total_seconds={:.3} seconds_per_sweep={:.3}",
        sweeps,
        b_seconds,
        b_seconds / sweeps as f64
    );
    println!("precomputed_jacobi_deltas={}", format_deltas(&deltas_b));

    // --- Strategy C: precomputed successors, in-place Gauss-Seidel.
    let mut values_c = lut.values.clone();
    let mut deltas_c = Vec::new();
    let c_started = Instant::now();
    for _ in 0..sweeps {
        deltas_c.push(gauss_seidel_precomputed(&successors, &indices, &mut values_c));
    }
    let c_seconds = c_started.elapsed().as_secs_f64();
    println!(
        "precomputed_gauss_seidel sweeps={} total_seconds={:.3} seconds_per_sweep={:.3}",
        sweeps,
        c_seconds,
        c_seconds / sweeps as f64
    );
    println!("precomputed_gauss_seidel_deltas={}", format_deltas(&deltas_c));

    // Residual after the same number of sweeps, measured identically for each
    // strategy: one deterministic Jacobi residual pass, no values written.
    for (name, values) in [("ondemand_jacobi", &values_a), ("precomputed_jacobi", &values_b), ("precomputed_gauss_seidel", &values_c)] {
        let (_, residual) = jacobi_precomputed(&successors, &indices, values);
        println!("{name}_residual_after_{sweeps}_sweeps={residual:.6e}");
    }

    println!("peak_rss_gb={:.3}", peak_rss_bytes() as f64 / 1e9);
}

fn format_deltas(deltas: &[f64]) -> String {
    deltas.iter().map(|delta| format!("{delta:.3e}")).collect::<Vec<_>>().join(",")
}

/// Encoding round trips plus sampled successor validation.
fn validate_encoding(lut: &TrainingLut, samples: usize) {
    let mut rng = SplitMix64::new(0x9e37_79b9_7f4a_7c15);
    let mut checked = 0usize;
    let mut successors_checked = 0usize;
    while checked < samples {
        let global = rng.index(lut.keys.len());
        let key = lut.key_at_global(global);
        let game = lut.encoding.decode(key);
        // Round trip: decode then re-encode must return the same key, and the
        // key's score field must agree with the decoded position.
        assert_eq!(lut.encoding.scores(key), (game.light_score, game.dark_score));
        assert_eq!(lut.lookup_index(key), global, "key round trip failed");
        if game.finished {
            continue;
        }
        assert_eq!(lut.encoding.encode_symmetrical(&game), key, "encode(decode(key)) != key");
        // Sampled successors: every generated successor must be a terminal
        // position or a key present in the map.
        let mut moves = [0i8; 8];
        for (roll, &probability) in lut.rules.roll_probabilities().iter().enumerate() {
            if probability == 0.0 {
                continue;
            }
            let mut rolled = game.clone();
            let move_count = rolled.apply_roll(roll as u8, &mut moves);
            if move_count == 0 {
                if !rolled.finished {
                    lut.lookup_index(lut.encoding.encode_symmetrical(&rolled));
                }
                successors_checked += 1;
                continue;
            }
            for &source in &moves[..move_count] {
                let mut next = rolled.clone();
                next.apply_move(source, lut.rules);
                if !next.finished {
                    lut.lookup_index(lut.encoding.encode_symmetrical(&next));
                }
                successors_checked += 1;
            }
        }
        checked += 1;
    }
    println!("encoding_round_trips_checked={checked}");
    println!("sampled_successors_checked={successors_checked}");
}

// ---------------------------------------------------------------------------
// Agent evaluation: heuristics, exact move regret, and the stage curve.
//
// Everything here scores an agent against the exact solution, so no Monte Carlo
// noise enters the value estimates. Heuristics are written as drop-in
// replacements for the lookup table -- each returns a light-favouring score, the
// same shape as Lut::light_win_percent -- so one argmax routine serves them all.
// ---------------------------------------------------------------------------

const FEATURE_COUNT: usize = 14;
/// Every feature is paired self/opponent, so a weight vector can be made
/// antisymmetric. That matters: a position's value from the mover's view and
/// from the opponent's view are related by `v -> 100 - v`, so only an
/// antisymmetric score (about the 50 intercept) is consistent under that
/// reflection. An unpaired feature would silently bias every comparison between
/// a move that keeps the turn and one that passes it.
const FEATURE_NAMES: [&str; FEATURE_COUNT] = [
    "advancement_self", "advancement_opp", "scored_self", "scored_opp",
    "hand_self", "hand_opp", "safe_self", "safe_opp",
    "exposure_self", "threat_self", "centre_self", "centre_opp",
    "frontmost_self", "frontmost_opp",
];
/// Weight vectors carry a trailing intercept, so a score is on the 0-100
/// win-percentage scale and can be reflected the same way the table's values are.
const WEIGHT_COUNT: usize = FEATURE_COUNT + 1;

/// Features of a position, always from the perspective of the player to move.
///
/// Board pieces store their own path index (`sign * (path_index + 1)`), so
/// advancement is just the sum of the stored magnitudes.
fn features(lut: &Lut, game: &Game) -> [f64; FEATURE_COUNT] {
    let sign = game.turn_sign();
    let (self_path, opp_path) = if game.is_light_turn {
        (lut.rules.light_path(), lut.rules.dark_path())
    } else {
        (lut.rules.dark_path(), lut.rules.light_path())
    };

    let mut advancement_self = 0.0;
    let mut advancement_opp = 0.0;
    let mut frontmost_self: f64 = 0.0;
    let mut frontmost_opp: f64 = 0.0;
    for tile in 0..BOARD_LEN {
        let piece = game.board[tile];
        if piece == 0 {
            continue;
        }
        let progress = piece.abs() as f64;
        if piece * sign > 0 {
            advancement_self += progress;
            frontmost_self = frontmost_self.max(progress);
        } else {
            advancement_opp += progress;
            frontmost_opp = frontmost_opp.max(progress);
        }
    }

    // Private tiles: on one player's path only, so they cannot be captured.
    let (safe_self_tiles, safe_opp_tiles) = if game.is_light_turn {
        (&lut.encoding.light_safe_indices, &lut.encoding.dark_safe_indices)
    } else {
        (&lut.encoding.dark_safe_indices, &lut.encoding.light_safe_indices)
    };
    let count_on = |tiles: &Vec<usize>, want_self: bool| {
        tiles
            .iter()
            .filter(|&&tile| {
                let piece = game.board[tile];
                piece != 0 && ((piece * sign > 0) == want_self)
            })
            .count() as f64
    };

    let (scored_self, scored_opp) = if game.is_light_turn {
        (game.light_score, game.dark_score)
    } else {
        (game.dark_score, game.light_score)
    };
    let (hand_self, hand_opp) = if game.is_light_turn {
        (game.light_pieces, game.dark_pieces)
    } else {
        (game.dark_pieces, game.light_pieces)
    };

    let _ = (self_path, opp_path);
    [
        advancement_self,
        advancement_opp,
        scored_self as f64,
        scored_opp as f64,
        hand_self as f64,
        hand_opp as f64,
        count_on(safe_self_tiles, true),
        count_on(safe_opp_tiles, false),
        capture_probability(lut, game, false),
        capture_probability(lut, game, true),
        if game.board[CENTRE_ROSETTE] * sign > 0 { 1.0 } else { 0.0 },
        if game.board[CENTRE_ROSETTE] * sign < 0 { 1.0 } else { 0.0 },
        frontmost_self,
        frontmost_opp,
    ]
}

/// The contested rosette in the middle lane, board tile (2, 4).
const CENTRE_ROSETTE: usize = 10;

/// Probability that a capture is available on the next turn.
///
/// With `for_mover` false this is the exposure of the moving player's pieces:
/// the chance the opponent, moving next, can take one of them. With `for_mover`
/// true it is the moving player's own capture chances. Move generation is reused
/// rather than reimplemented, so the safe-rosette rule is honoured for free.
fn capture_probability(lut: &Lut, game: &Game, for_mover: bool) -> f64 {
    let mut probe = game.clone();
    if !for_mover {
        probe.is_light_turn = !probe.is_light_turn;
    }
    probe.roll = -1;
    let sign = probe.turn_sign();
    let path = if probe.is_light_turn { lut.rules.light_path() } else { lut.rules.dark_path() };

    let mut total = 0.0;
    let mut moves = [0i8; 8];
    for (roll, &probability) in lut.rules.roll_probabilities().iter().enumerate() {
        if probability == 0.0 {
            continue;
        }
        let mut rolled = probe.clone();
        let count = rolled.apply_roll(roll as u8, &mut moves);
        let captures = moves[..count].iter().any(|&source| {
            let destination = source as isize + roll as isize;
            if destination < 0 || destination as usize >= path.len() {
                return false;
            }
            let occupant = rolled.board[path[destination as usize]];
            occupant != 0 && occupant * sign < 0
        });
        if captures {
            total += probability;
        }
    }
    total
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Heuristic {
    Random,
    Advancement,
    Lead,
    ScoreRace,
    Safety,
    Centre,
    Exposure,
    Composite,
}

impl Heuristic {
    fn name(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Advancement => "advancement",
            Self::Lead => "lead",
            Self::ScoreRace => "score_race",
            Self::Safety => "safety",
            Self::Centre => "centre",
            Self::Exposure => "exposure",
            Self::Composite => "composite",
        }
    }

    fn all() -> &'static [Heuristic] {
        &[
            Self::Random, Self::Advancement, Self::Lead, Self::ScoreRace,
            Self::Safety, Self::Centre, Self::Exposure, Self::Composite,
        ]
    }

    /// Weights over FEATURE_NAMES. Hand-set to make each rung of the ladder
    /// isolate one idea; the fitted model replaces these with regression on the
    /// exact values.
    /// Weights over FEATURE_NAMES plus a trailing intercept. Each is written
    /// antisymmetrically (every self term mirrored by its opponent term) so the
    /// score behaves correctly under the `v -> 100 - v` reflection.
    fn weights(self, path_len: f64) -> [f64; WEIGHT_COUNT] {
        let mut w = [0.0; WEIGHT_COUNT];
        w[FEATURE_COUNT] = 50.0; // intercept: an even position scores 50
        fn set(w: &mut [f64; WEIGHT_COUNT], index: usize, value: f64) {
            w[index] = value;
            w[index + 1] = -value;
        }
        match self {
            Self::Random => {}
            Self::Advancement => set(&mut w, 0, 1.0),
            Self::Lead => {
                set(&mut w, 0, 1.0);
                set(&mut w, 4, -1.0); // pieces still in hand are behind
            }
            Self::ScoreRace => {
                set(&mut w, 0, 1.0);
                // A scored piece is safe forever, so it outweighs the
                // advancement it represents.
                set(&mut w, 2, 2.0 * path_len);
            }
            Self::Safety => {
                set(&mut w, 0, 1.0);
                set(&mut w, 2, 2.0 * path_len);
                set(&mut w, 6, 2.0);
            }
            Self::Centre => {
                set(&mut w, 0, 1.0);
                set(&mut w, 2, 2.0 * path_len);
                set(&mut w, 6, 2.0);
                set(&mut w, 10, 4.0);
            }
            Self::Exposure => {
                set(&mut w, 0, 1.0);
                set(&mut w, 2, 2.0 * path_len);
                set(&mut w, 6, 2.0);
                set(&mut w, 10, 4.0);
                // exposure_self is the opponent's threat, so the pair (8, 9) is
                // already antisymmetric with a single sign.
                w[8] = -6.0;
                w[9] = 6.0;
            }
            Self::Composite => {
                set(&mut w, 0, 1.0);
                set(&mut w, 2, 2.0 * path_len);
                set(&mut w, 6, 2.0);
                set(&mut w, 10, 4.0);
                w[8] = -6.0;
                w[9] = 6.0;
                set(&mut w, 12, 0.5);
            }
        }
        w
    }
}

/// A light-favouring score for a position, mirroring Lut::light_win_percent so
/// heuristics can be swapped in wherever the table is used.
fn heuristic_light_value(
    weights: Option<&[f64; WEIGHT_COUNT]>,
    lut: &Lut,
    game: &Game,
    rng: &mut SplitMix64,
) -> f64 {
    if game.finished {
        return if game.light_score >= lut.rules.pieces() { f64::INFINITY } else { f64::NEG_INFINITY };
    }
    let Some(weights) = weights else {
        // Uniform noise, so argmax picks a uniformly random legal move.
        return (rng.next_u64() >> 11) as f64;
    };
    let features = features(lut, game);
    // features() is from the mover's perspective, so this estimates the mover's
    // win percentage; convert to light's exactly as Lut::light_win_percent does.
    // A negation would be wrong: the two perspectives are related by a
    // reflection about 50, and the difference is not constant across successors
    // because a move onto a rosette keeps the turn while others pass it.
    let mover: f64 = features
        .iter()
        .zip(weights.iter())
        .map(|(f, w)| f * w)
        .sum::<f64>()
        + weights[FEATURE_COUNT];
    if game.is_light_turn { mover } else { 100.0 - mover }
}

/// Pick a move with an arbitrary evaluator, mirroring choose_optimal_move.
fn choose_move_with(
    weights: Option<&[f64; WEIGHT_COUNT]>,
    lut: &Lut,
    game: &Game,
    moves: &[i8],
    rng: &mut SplitMix64,
) -> i8 {
    assert!(!moves.is_empty());
    if moves.len() == 1 {
        return moves[0];
    }
    let light_turn = game.is_light_turn;
    let mut best_move = moves[0];
    let mut best = if light_turn { f64::NEG_INFINITY } else { f64::INFINITY };
    for &source in moves {
        let mut next = game.clone();
        next.apply_move(source, lut.rules);
        let value = heuristic_light_value(weights, lut, &next, rng);
        let better = if light_turn { value > best } else { value < best };
        if better {
            best = value;
            best_move = source;
        }
    }
    best_move
}

/// Dump every candidate move of every sampled position: the successor's
/// features, whether the move passed the turn, and the successor's exact value
/// from the *original* mover's perspective.
///
/// This makes the whole move-choice problem solvable offline. For any weight
/// vector the score of a successor is `f . w + b`, reflected to `100 - (f . w + b)`
/// when the move passed the turn, so regret can be evaluated for arbitrary
/// weights without re-running the engine. That is what makes a Shapley
/// decomposition over regret affordable: 2^14 subsets would be hopeless if each
/// needed a fresh pass through move generation.
fn dump_moves(model: &Path, output: &Path, samples: usize, on_policy: bool, seed: u64) {
    let lut = Lut::read(model);
    let mut rng = SplitMix64::new(seed);
    let mut file = BufWriter::new(File::create(output).unwrap());
    writeln!(file, "state,move,turn_passed,value_mover,{}", FEATURE_NAMES.join(",")).unwrap();

    let mut moves = [0i8; 8];
    let mut written = 0usize;
    let mut state_id = 0usize;
    let mut game = Game::initial(lut.rules);

    while written < samples {
        // Either walk an optimal game or jump to a random stored position.
        let decision = if on_policy {
            if game.finished {
                game = Game::initial(lut.rules);
            }
            let roll = lut.rules.roll(&mut rng);
            let count = game.apply_roll(roll, &mut moves);
            if count == 0 {
                continue;
            }
            let chosen = choose_optimal_move(&lut, &game, &moves[..count]);
            let snapshot = game.clone();
            game.apply_move(chosen, lut.rules);
            if count < 2 {
                continue;
            }
            snapshot
        } else {
            let (key, _) = lut.key_value_at_global(rng.index(lut.total));
            let mut candidate = lut.encoding.decode(key);
            if candidate.finished {
                continue;
            }
            let roll = lut.rules.roll(&mut rng);
            let count = candidate.apply_roll(roll, &mut moves);
            if count < 2 {
                continue;
            }
            candidate
        };

        let count = decision.available_moves(&mut moves);
        let mover_is_light = decision.is_light_turn;
        for (move_index, &source) in moves[..count].iter().enumerate() {
            let mut next = decision.clone();
            next.apply_move(source, lut.rules);
            let turn_passed = next.is_light_turn != mover_is_light;
            let light = lut.light_win_percent(&next);
            let value_mover = if mover_is_light { light } else { 100.0 - light };
            let row = if next.finished {
                // A finished position has no features; its value is decisive, so
                // mark it and let the consumer handle it.
                vec!["0".to_string(); FEATURE_COUNT]
            } else {
                features(&lut, &next).iter().map(|v| format!("{v:.4}")).collect()
            };
            writeln!(
                file,
                "{state_id},{move_index},{},{value_mover:.9},{}",
                u8::from(turn_passed),
                row.join(",")
            )
            .unwrap();
        }
        written += 1;
        state_id += 1;
    }
    file.flush().unwrap();
    eprintln!("wrote {written} positions to {}", output.display());
}

/// Accumulate the normal equations for a least-squares fit over **every**
/// non-terminal state in the map, rather than a sample.
///
/// Least squares only needs `X'X` and `X'y`, which are sums over rows, so the
/// exact full-population fit costs one streaming pass and a fixed-size
/// accumulator -- no need to hold 500 million feature rows anywhere. Since the
/// map is the entire population of states, there is no sampling error and
/// nothing to overfit to.
///
/// A further consequence: every subset of features can afterwards be fitted by
/// solving the corresponding submatrix of this one Gram matrix, so all 2^k
/// subset fits come free from a single pass.
fn feature_gram(model: &Path, output: &Path) {
    let lut = Lut::read(model);
    let columns = FEATURE_COUNT + 1; // features plus the intercept column
    let threads = thread::available_parallelism().map(usize::from).unwrap_or(1).max(1);
    let chunk = (lut.total + threads - 1) / threads;
    let started = Instant::now();

    let partials: Vec<(Vec<f64>, Vec<f64>, f64, usize)> = thread::scope(|scope| {
        let mut handles = Vec::new();
        for thread_index in 0..threads {
            let begin = thread_index * chunk;
            let end = (begin + chunk).min(lut.total);
            let lut = &lut;
            handles.push(scope.spawn(move || {
                let mut xtx = vec![0.0f64; columns * columns];
                let mut xty = vec![0.0f64; columns];
                let mut yty = 0.0f64;
                let mut rows = 0usize;
                let mut buffer = vec![0.0f64; columns];
                for global in begin..end {
                    let (key, _) = lut.key_value_at_global(global);
                    let game = lut.encoding.decode(key);
                    if game.finished {
                        continue;
                    }
                    let f = features(lut, &game);
                    buffer[..FEATURE_COUNT].copy_from_slice(&f);
                    buffer[FEATURE_COUNT] = 1.0;
                    // The stored value is the mover's win percentage, which is
                    // the perspective features() uses.
                    let light = lut.light_win_percent(&game);
                    let y = if game.is_light_turn { light } else { 100.0 - light };
                    for i in 0..columns {
                        let bi = buffer[i];
                        if bi == 0.0 {
                            continue;
                        }
                        xty[i] += bi * y;
                        for j in i..columns {
                            xtx[i * columns + j] += bi * buffer[j];
                        }
                    }
                    yty += y * y;
                    rows += 1;
                }
                (xtx, xty, yty, rows)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut xtx = vec![0.0f64; columns * columns];
    let mut xty = vec![0.0f64; columns];
    let mut yty = 0.0f64;
    let mut rows = 0usize;
    for (a, b, c, n) in partials {
        for i in 0..columns * columns {
            xtx[i] += a[i];
        }
        for i in 0..columns {
            xty[i] += b[i];
        }
        yty += c;
        rows += n;
    }
    // Only the upper triangle was accumulated.
    for i in 0..columns {
        for j in 0..i {
            xtx[i * columns + j] = xtx[j * columns + i];
        }
    }

    let mut file = BufWriter::new(File::create(output).unwrap());
    writeln!(file, "# exact normal equations over all non-terminal states").unwrap();
    writeln!(file, "# columns: {},intercept", FEATURE_NAMES.join(",")).unwrap();
    writeln!(file, "rows,{rows}").unwrap();
    writeln!(file, "yty,{yty:.10e}").unwrap();
    for i in 0..columns {
        let row: Vec<String> = (0..columns).map(|j| format!("{:.10e}", xtx[i * columns + j])).collect();
        writeln!(file, "xtx,{}", row.join(",")).unwrap();
    }
    let row: Vec<String> = xty.iter().map(|v| format!("{v:.10e}")).collect();
    writeln!(file, "xty,{}", row.join(",")).unwrap();
    file.flush().unwrap();
    eprintln!(
        "accumulated {rows} states in {:.1}s -> {}",
        started.elapsed().as_secs_f64(),
        output.display()
    );
}

const MOVE_FEATURE_COUNT: usize = 16;
const MOVE_FEATURE_NAMES: [&str; MOVE_FEATURE_COUNT] = [
    "advance", "captures", "scores", "enters", "lands_rosette", "lands_centre",
    "leaves_centre", "dest_safe", "src_was_exposed", "delta_exposure",
    "delta_threat", "keeps_turn",
    // Magnitudes, not just occurrence. Error analysis showed the binary
    // versions cannot distinguish taking a piece one square from home from
    // taking one that just entered, and capture decisions carry by far the most
    // regret.
    "capture_value", "rescue_value", "delta_exposure_value", "delta_threat_value",
];

/// Expected progress captured on the next turn, assuming the capturing player
/// takes the most advanced piece available to them.
///
/// This is the magnitude behind `capture_probability`: a 1-in-4 chance of taking
/// a piece 13 squares along is worth far more than the same chance at a piece
/// that has just entered.
fn weighted_capture_value(lut: &Lut, game: &Game, for_mover: bool) -> f64 {
    let mut probe = game.clone();
    if !for_mover {
        probe.is_light_turn = !probe.is_light_turn;
    }
    probe.roll = -1;
    let sign = probe.turn_sign();
    let path = if probe.is_light_turn { lut.rules.light_path() } else { lut.rules.dark_path() };

    let mut total = 0.0;
    let mut moves = [0i8; 8];
    for (roll, &probability) in lut.rules.roll_probabilities().iter().enumerate() {
        if probability == 0.0 {
            continue;
        }
        let mut rolled = probe.clone();
        let count = rolled.apply_roll(roll as u8, &mut moves);
        let mut best = 0.0f64;
        for &source in &moves[..count] {
            let destination = source as isize + roll as isize;
            if destination < 0 || destination as usize >= path.len() {
                continue;
            }
            let occupant = rolled.board[path[destination as usize]];
            if occupant != 0 && occupant * sign < 0 {
                best = best.max(occupant.abs() as f64);
            }
        }
        total += probability * best;
    }
    total
}

/// Features of the *move* rather than of the resulting position.
///
/// The within-position centring result says only what differs between sibling
/// moves can affect which is chosen. Move features are that difference by
/// construction: every one of them varies across the candidates of a position,
/// where a state feature like `scored_self` is usually identical for all of
/// them. `captures`, `scores` and `enters` in particular are properties of the
/// transition that no function of the successor position alone recovers.
fn move_features(
    lut: &Lut,
    rolled: &Game,
    source: i8,
    roll: usize,
    next: &Game,
) -> [f64; MOVE_FEATURE_COUNT] {
    let mover_is_light = rolled.is_light_turn;
    let sign = rolled.turn_sign();
    let path = if mover_is_light { lut.rules.light_path() } else { lut.rules.dark_path() };
    let destination_index = source as isize + roll as isize;
    let scores = destination_index >= path.len() as isize;
    let destination = if scores { usize::MAX } else { path[destination_index as usize] };

    let captures = !scores && {
        let occupant = rolled.board[destination];
        occupant != 0 && occupant * sign < 0
    };
    let opponent_path = if mover_is_light { lut.rules.dark_path() } else { lut.rules.light_path() };
    let destination_safe = !scores && !opponent_path.contains(&destination);
    let turn_passed = next.is_light_turn != mover_is_light;

    // Exposure of the mover's pieces, and the mover's own capture chances,
    // after the move. Which argument to capture_probability depends on whose
    // turn it now is.
    let exposure_before = capture_probability(lut, rolled, false);
    let threat_before = capture_probability(lut, rolled, true);
    let exposure_value_before = weighted_capture_value(lut, rolled, false);
    let threat_value_before = weighted_capture_value(lut, rolled, true);
    let (exposure_after, threat_after, exposure_value_after, threat_value_after) = if next.finished {
        (0.0, 0.0, 0.0, 0.0)
    } else {
        (
            capture_probability(lut, next, turn_passed),
            capture_probability(lut, next, !turn_passed),
            weighted_capture_value(lut, next, turn_passed),
            weighted_capture_value(lut, next, !turn_passed),
        )
    };

    // How advanced was the piece taken, and the piece rescued.
    let capture_value = if captures { rolled.board[destination].abs() as f64 } else { 0.0 };
    let rescue_value = if source >= 0 && source_was_exposed_check(lut, rolled, path[source as usize]) {
        (source as f64) + 1.0
    } else {
        0.0
    };

    let source_tile = if source >= 0 { Some(path[source as usize]) } else { None };
    let source_was_exposed = source_tile
        .map(|tile| opponent_path.contains(&tile) && !(lut.rules.safe_rosettes() && ROSETTES.contains(&tile)))
        .unwrap_or(false);

    [
        roll as f64,
        f64::from(captures),
        f64::from(scores),
        f64::from(source < 0),
        f64::from(!scores && ROSETTES.contains(&destination)),
        f64::from(!scores && destination == CENTRE_ROSETTE),
        f64::from(source_tile == Some(CENTRE_ROSETTE)),
        f64::from(destination_safe),
        f64::from(source_was_exposed),
        exposure_after - exposure_before,
        threat_after - threat_before,
        f64::from(!turn_passed),
        capture_value,
        rescue_value,
        exposure_value_after - exposure_value_before,
        threat_value_after - threat_value_before,
    ]
}

/// Whether a tile is one the opponent can actually reach and take.
fn source_was_exposed_check(lut: &Lut, game: &Game, tile: usize) -> bool {
    let opponent_path = if game.is_light_turn { lut.rules.dark_path() } else { lut.rules.light_path() };
    opponent_path.contains(&tile)
        && !(lut.rules.safe_rosettes() && ROSETTES.contains(&tile))
}

/// Dump every candidate move with BOTH state features and move features, plus
/// its exact value, so competing policy families can be fitted and compared
/// offline on identical data.
fn dump_move_features(model: &Path, output: &Path, samples: usize, seed: u64) {
    let lut = Lut::read(model);
    let mut rng = SplitMix64::new(seed);
    let mut file = BufWriter::new(File::create(output).unwrap());
    writeln!(
        file,
        "state,move,turn_passed,value_mover,occupancy,{},{}",
        FEATURE_NAMES.join(","),
        MOVE_FEATURE_NAMES.join(",")
    )
    .unwrap();

    let mut moves = [0i8; 8];
    let mut game = Game::initial(lut.rules);
    let mut positions = 0usize;
    while positions < samples {
        if game.finished {
            game = Game::initial(lut.rules);
        }
        let roll = lut.rules.roll(&mut rng) as usize;
        let count = game.apply_roll(roll as u8, &mut moves);
        if count == 0 {
            continue;
        }
        let snapshot = game.clone();
        let chosen = choose_optimal_move(&lut, &game, &moves[..count]);
        game.apply_move(chosen, lut.rules);
        if count < 2 {
            continue;
        }

        let mover_is_light = snapshot.is_light_turn;
        for (index, &source) in moves[..count].iter().enumerate() {
            let mut next = snapshot.clone();
            next.apply_move(source, lut.rules);
            let passed = next.is_light_turn != mover_is_light;
            let light = lut.light_win_percent(&next);
            let value = if mover_is_light { light } else { 100.0 - light };
            let state_row: Vec<String> = if next.finished {
                vec!["0".into(); FEATURE_COUNT]
            } else {
                features(&lut, &next).iter().map(|v| format!("{v:.4}")).collect()
            };
            let move_row: Vec<String> = move_features(&lut, &snapshot, source, roll, &next)
                .iter()
                .map(|v| format!("{v:.4}"))
                .collect();
            // Occupancy along the mover's own path, one character per path
            // position: 0 empty, 1 the player to move in `next`, 2 the other.
            // This is what an N-tuple network indexes into, and it is written
            // from the same perspective as the features so the reflection
            // handles both consistently.
            let occupancy: String = {
                let path = if next.is_light_turn {
                    lut.rules.light_path()
                } else {
                    lut.rules.dark_path()
                };
                let sign = if next.is_light_turn { 1i8 } else { -1i8 };
                path.iter()
                    .map(|&tile| {
                        let piece = next.board[tile];
                        if piece == 0 {
                            '0'
                        } else if piece * sign > 0 {
                            '1'
                        } else {
                            '2'
                        }
                    })
                    .collect()
            };
            writeln!(
                file,
                "{positions},{index},{},{value:.9},{occupancy},{},{}",
                u8::from(passed),
                state_row.join(","),
                move_row.join(",")
            )
            .unwrap();
        }
        positions += 1;
    }
    file.flush().unwrap();
    eprintln!("wrote {positions} positions to {}", output.display());
}

/// Expectimax over dice, with a heuristic at the leaves.
///
/// Chance nodes are the rolls, so this averages over rolls and takes the best
/// (or worst, for the opponent) move at each decision. Depth counts plies of
/// move choice; depth 0 evaluates the position directly, which is what the
/// 1-ply greedy policy does.
fn expectimax_light(
    weights: Option<&[f64; WEIGHT_COUNT]>,
    lut: &Lut,
    game: &Game,
    depth: usize,
    rng: &mut SplitMix64,
) -> f64 {
    if game.finished {
        return if game.light_score >= lut.rules.pieces() { 100.0 } else { 0.0 };
    }
    if depth == 0 {
        return heuristic_light_value(weights, lut, game, rng);
    }
    let mut moves = [0i8; 8];
    let mut total = 0.0;
    for (roll, &probability) in lut.rules.roll_probabilities().iter().enumerate() {
        if probability == 0.0 {
            continue;
        }
        let mut rolled = game.clone();
        let count = rolled.apply_roll(roll as u8, &mut moves);
        let value = if count == 0 {
            // No legal move: the turn simply passes, which is not a decision, so
            // it does not consume a ply.
            expectimax_light(weights, lut, &rolled, depth - 1, rng)
        } else {
            let light_turn = rolled.is_light_turn;
            let mut best = if light_turn { f64::NEG_INFINITY } else { f64::INFINITY };
            for &source in &moves[..count] {
                let mut next = rolled.clone();
                next.apply_move(source, lut.rules);
                let value = expectimax_light(weights, lut, &next, depth - 1, rng);
                if light_turn {
                    if value > best {
                        best = value;
                    }
                } else if value < best {
                    best = value;
                }
            }
            best
        };
        total += probability * value;
    }
    total
}

/// Regret of a weighted heuristic as a function of search depth.
///
/// Depth 1 is the greedy policy used everywhere else: evaluate each successor
/// with the heuristic. Deeper searches average over the opponent's roll and
/// reply before evaluating, so a weak evaluator can compensate with lookahead.
fn depth_regret(
    model: &Path,
    output: &Path,
    samples: usize,
    max_depth: usize,
    weights_path: Option<&Path>,
    seed: u64,
) {
    let lut = Lut::read(model);
    let weights = weights_path.map(|path| {
        let text = fs::read_to_string(path).expect("failed to read weights");
        let values: Vec<f64> = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| line.parse::<f64>().expect("weights must be one float per line"))
            .collect();
        assert_eq!(values.len(), WEIGHT_COUNT, "expected {WEIGHT_COUNT} weights");
        let mut array = [0.0; WEIGHT_COUNT];
        array.copy_from_slice(&values);
        array
    });

    // Sample decision states from optimal play.
    let mut rng = SplitMix64::new(seed);
    let mut states = Vec::with_capacity(samples);
    let mut moves = [0i8; 8];
    let mut game = Game::initial(lut.rules);
    while states.len() < samples {
        if game.finished {
            game = Game::initial(lut.rules);
        }
        let roll = lut.rules.roll(&mut rng);
        let count = game.apply_roll(roll, &mut moves);
        if count == 0 {
            continue;
        }
        if count > 1 {
            states.push(game.clone());
        }
        let source = choose_optimal_move(&lut, &game, &moves[..count]);
        game.apply_move(source, lut.rules);
    }

    let mut file = BufWriter::new(File::create(output).unwrap());
    writeln!(file, "heuristic,depth,states,mean_regret,agreement_pct,seconds").unwrap();

    let named: Vec<(String, Option<[f64; WEIGHT_COUNT]>)> = {
        let mut list: Vec<(String, Option<[f64; WEIGHT_COUNT]>)> = vec![
            ("advancement".into(), Some(Heuristic::Advancement.weights(lut.rules.light_path().len() as f64))),
            ("composite".into(), Some(Heuristic::Composite.weights(lut.rules.light_path().len() as f64))),
        ];
        if let Some(w) = weights {
            list.push(("fitted".into(), Some(w)));
        }
        list
    };

    let threads = thread::available_parallelism().map(usize::from).unwrap_or(1).max(1);
    for (name, weight) in &named {
        for depth in 1..=max_depth {
            let started = Instant::now();
            let chunk = (states.len() + threads - 1) / threads;
            let (regret_sum, agree): (f64, usize) = thread::scope(|scope| {
                let mut handles = Vec::new();
                for part in states.chunks(chunk.max(1)) {
                    let lut = &lut;
                    handles.push(scope.spawn(move || {
                        let mut rng = SplitMix64::new(0x9e37_79b9);
                        let mut moves = [0i8; 8];
                        let mut sum = 0.0;
                        let mut agree = 0usize;
                        for game in part {
                            let count = game.available_moves(&mut moves);
                            let best = choose_optimal_move(lut, game, &moves[..count]);
                            let light_turn = game.is_light_turn;
                            let mut picked = moves[0];
                            let mut best_score =
                                if light_turn { f64::NEG_INFINITY } else { f64::INFINITY };
                            for &source in &moves[..count] {
                                let mut next = game.clone();
                                next.apply_move(source, lut.rules);
                                let value = expectimax_light(
                                    weight.as_ref(), lut, &next, depth - 1, &mut rng,
                                );
                                let better = if light_turn { value > best_score } else { value < best_score };
                                if better {
                                    best_score = value;
                                    picked = source;
                                }
                            }
                            let value_of = |source: i8| {
                                let mut next = game.clone();
                                next.apply_move(source, lut.rules);
                                let light = lut.light_win_percent(&next);
                                if light_turn { light } else { 100.0 - light }
                            };
                            let regret = (value_of(best) - value_of(picked)).max(0.0);
                            if regret == 0.0 {
                                agree += 1;
                            }
                            sum += regret;
                        }
                        (sum, agree)
                    }));
                }
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .fold((0.0, 0), |acc, item| (acc.0 + item.0, acc.1 + item.1))
            });
            let mean = regret_sum / states.len() as f64;
            let agreement = 100.0 * agree as f64 / states.len() as f64;
            let seconds = started.elapsed().as_secs_f64();
            writeln!(
                file,
                "{name},{depth},{},{mean:.6},{agreement:.4},{seconds:.2}",
                states.len()
            )
            .unwrap();
            eprintln!("{name:>12} depth={depth}: mean_regret={mean:.4} agreement={agreement:.2}% ({seconds:.1}s)");
        }
    }
    file.flush().unwrap();
}

/// Accumulate the *within-position centred* normal equations over every
/// decision in the map: every stored state, every roll that offers a choice.
///
/// This is the ordering counterpart of `feature_gram`. Subtracting each
/// position's mean from the design and target annihilates whatever is constant
/// across that position's sibling moves -- exactly the part that cannot affect
/// which move is chosen -- so the resulting fit targets move ordering rather
/// than value accuracy.
///
/// Reflection is folded in linearly. With `s = 1 - 2 * passed`, a successor's
/// mover-relative score is `s * (f . w) + s * b + 100 * passed`, so signing the
/// features by `s`, appending `s` as the intercept column, and moving the known
/// `100 * passed` to the target keeps the whole thing least squares.
///
/// Each decision is weighted by its roll probability, so a position that arises
/// from a 1-in-16 roll does not count the same as one from a 6-in-16 roll.
fn ordering_gram(model: &Path, output: &Path, stride: usize) {
    let lut = Lut::read(model);
    let columns = FEATURE_COUNT + 1;
    let threads = thread::available_parallelism().map(usize::from).unwrap_or(1).max(1);
    let chunk = (lut.total + threads - 1) / threads;
    let started = Instant::now();

    let partials: Vec<(Vec<f64>, Vec<f64>, f64, usize, f64)> = thread::scope(|scope| {
        let mut handles = Vec::new();
        for thread_index in 0..threads {
            let begin = thread_index * chunk;
            let end = (begin + chunk).min(lut.total);
            let lut = &lut;
            handles.push(scope.spawn(move || {
                let mut xtx = vec![0.0f64; columns * columns];
                let mut xty = vec![0.0f64; columns];
                let mut yty = 0.0f64;
                let mut decisions = 0usize;
                let mut weight_total = 0.0f64;

                let mut moves = [0i8; 8];
                let mut design = vec![0.0f64; 8 * columns];
                let mut targets = [0.0f64; 8];

                let mut global = begin;
                while global < end {
                    let (key, _) = lut.key_value_at_global(global);
                    global += stride;
                    let base = lut.encoding.decode(key);
                    if base.finished {
                        continue;
                    }
                    for (roll, &probability) in lut.rules.roll_probabilities().iter().enumerate() {
                        if probability == 0.0 {
                            continue;
                        }
                        let mut rolled = base.clone();
                        let count = rolled.apply_roll(roll as u8, &mut moves);
                        if count < 2 {
                            continue; // no choice to make, so nothing to order
                        }
                        let mover_is_light = rolled.is_light_turn;
                        for (index, &source) in moves[..count].iter().enumerate() {
                            let mut next = rolled.clone();
                            next.apply_move(source, lut.rules);
                            let passed = next.is_light_turn != mover_is_light;
                            let sign = if passed { -1.0 } else { 1.0 };
                            let light = lut.light_win_percent(&next);
                            let value = if mover_is_light { light } else { 100.0 - light };
                            let row = &mut design[index * columns..(index + 1) * columns];
                            if next.finished {
                                row[..FEATURE_COUNT].fill(0.0);
                            } else {
                                for (slot, value) in
                                    row[..FEATURE_COUNT].iter_mut().zip(features(lut, &next))
                                {
                                    *slot = sign * value;
                                }
                            }
                            row[FEATURE_COUNT] = sign;
                            targets[index] = value - if passed { 100.0 } else { 0.0 };
                        }

                        // Centre within this position.
                        let inverse = 1.0 / count as f64;
                        let mut mean_row = vec![0.0f64; columns];
                        let mut mean_target = 0.0f64;
                        for index in 0..count {
                            for column in 0..columns {
                                mean_row[column] += design[index * columns + column] * inverse;
                            }
                            mean_target += targets[index] * inverse;
                        }
                        for index in 0..count {
                            let row = &design[index * columns..(index + 1) * columns];
                            let y = targets[index] - mean_target;
                            for i in 0..columns {
                                let a = row[i] - mean_row[i];
                                if a == 0.0 {
                                    continue;
                                }
                                xty[i] += probability * a * y;
                                for j in i..columns {
                                    xtx[i * columns + j] += probability * a * (row[j] - mean_row[j]);
                                }
                            }
                            yty += probability * y * y;
                        }
                        decisions += 1;
                        weight_total += probability;
                    }
                }
                (xtx, xty, yty, decisions, weight_total)
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut xtx = vec![0.0f64; columns * columns];
    let mut xty = vec![0.0f64; columns];
    let mut yty = 0.0f64;
    let mut decisions = 0usize;
    let mut weight_total = 0.0f64;
    for (a, b, c, n, w) in partials {
        for i in 0..columns * columns {
            xtx[i] += a[i];
        }
        for i in 0..columns {
            xty[i] += b[i];
        }
        yty += c;
        decisions += n;
        weight_total += w;
    }
    for i in 0..columns {
        for j in 0..i {
            xtx[i * columns + j] = xtx[j * columns + i];
        }
    }

    let mut file = BufWriter::new(File::create(output).unwrap());
    writeln!(file, "# within-position centred normal equations over every decision").unwrap();
    writeln!(file, "# columns: {},intercept", FEATURE_NAMES.join(",")).unwrap();
    writeln!(file, "rows,{decisions}").unwrap();
    writeln!(file, "weight,{weight_total:.10e}").unwrap();
    writeln!(file, "yty,{yty:.10e}").unwrap();
    for i in 0..columns {
        let row: Vec<String> = (0..columns).map(|j| format!("{:.10e}", xtx[i * columns + j])).collect();
        writeln!(file, "xtx,{}", row.join(",")).unwrap();
    }
    let row: Vec<String> = xty.iter().map(|v| format!("{v:.10e}")).collect();
    writeln!(file, "xty,{}", row.join(",")).unwrap();
    file.flush().unwrap();
    eprintln!(
        "accumulated {decisions} decisions (stride {stride}) in {:.1}s -> {}",
        started.elapsed().as_secs_f64(),
        output.display()
    );
}

/// Index of a state's score layer in the order the solver completes them.
fn layer_index_of(lut: &Lut, game: &Game, pair_to_index: &[usize]) -> usize {
    let low = game.light_score.min(game.dark_score) as usize;
    let high = game.light_score.max(game.dark_score) as usize;
    let _ = lut;
    pair_to_index[low * 8 + high]
}

/// Play one game where `stage_agent` plays optimally only once the position has
/// reached score layer `stage` or later in solve order, and uniformly at random
/// before that. The opponent always plays optimally. Returns true if the stage
/// agent won.
fn play_stage_game(
    lut: &Lut,
    stage: usize,
    stage_agent_is_light: bool,
    pair_to_index: &[usize],
    rng: &mut SplitMix64,
) -> bool {
    let mut game = Game::initial(lut.rules);
    let mut moves = [0i8; 8];
    let mut plies = 0usize;
    while !game.finished {
        plies += 1;
        assert!(plies < 100_000, "game failed to terminate");
        let roll = lut.rules.roll(rng);
        let count = game.apply_roll(roll, &mut moves);
        if count == 0 {
            continue;
        }
        let stage_agent_to_move = game.is_light_turn == stage_agent_is_light;
        // Layers are solved from high scores down, so a layer index below
        // `stage` is one the agent would already have solved.
        let solved_here = layer_index_of(lut, &game, pair_to_index) < stage;
        let source = if stage_agent_to_move && !solved_here {
            moves[rng.index(count)]
        } else {
            choose_optimal_move(lut, &game, &moves[..count])
        };
        game.apply_move(source, lut.rules);
    }
    (game.light_score >= lut.rules.pieces()) == stage_agent_is_light
}

/// Stage 0 of the analysis roadmap: how much of the game's difficulty lives in
/// each score layer. For every stage k, an agent that plays randomly until the
/// position reaches layer k and optimally thereafter is scored against a fully
/// optimal opponent, alternating sides so a converged agent sits at 50%.
fn stage_curve(model: &Path, output: &Path, games_per_stage: usize, seed: u64) {
    let lut = Lut::read(model);
    let pairs = score_pairs(lut.rules.pieces());
    let mut pair_to_index = vec![usize::MAX; 64];
    for (index, &(low, high)) in pairs.iter().enumerate() {
        pair_to_index[low as usize * 8 + high as usize] = index;
    }

    let threads = thread::available_parallelism().map(usize::from).unwrap_or(1).max(1);
    let mut output_file = BufWriter::new(File::create(output).unwrap());
    writeln!(
        output_file,
        "stage,layer_min,layer_max,layer_states,cumulative_states,games,wins,win_pct"
    )
    .unwrap();

    // Layer sizes give a self-contained x axis (states solved so far). Multiply
    // by per-layer sweep counts from a training log to get state expansions.
    let mut layer_states = vec![0usize; pairs.len()];
    let started = Instant::now();
    for global in 0..lut.total {
        let (key, _) = lut.key_value_at_global(global);
        let (light, dark) = lut.encoding.scores(key);
        if light >= lut.rules.pieces() || dark >= lut.rules.pieces() {
            continue;
        }
        let index = pair_to_index[light.min(dark) as usize * 8 + light.max(dark) as usize];
        if index != usize::MAX {
            layer_states[index] += 1;
        }
    }
    eprintln!("counted layer sizes in {:.1}s", started.elapsed().as_secs_f64());

    let mut cumulative = 0usize;
    for (stage, &pair) in pairs.iter().enumerate() {
        cumulative += layer_states[stage];
        // stage k means "layers with index < k are solved", so evaluate the
        // agent that has completed stages 0..=stage.
        let solved_through = stage + 1;
        let chunk = (games_per_stage + threads - 1) / threads;
        let wins: usize = thread::scope(|scope| {
            let mut handles = Vec::new();
            let mut remaining = games_per_stage;
            for thread_index in 0..threads {
                let count = chunk.min(remaining);
                remaining -= count;
                let pair_to_index = &pair_to_index;
                let lut = &lut;
                handles.push(scope.spawn(move || {
                    let mut rng = SplitMix64::new(
                        seed ^ ((stage as u64) << 32)
                            ^ (thread_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                    );
                    let mut wins = 0usize;
                    for game_index in 0..count {
                        // Alternate sides so the ceiling is exactly 50%.
                        let as_light = game_index % 2 == 0;
                        if play_stage_game(lut, solved_through, as_light, pair_to_index, &mut rng) {
                            wins += 1;
                        }
                    }
                    wins
                }));
            }
            handles.into_iter().map(|h| h.join().unwrap()).sum()
        });
        let percent = 100.0 * wins as f64 / games_per_stage as f64;
        writeln!(
            output_file,
            "{stage},{},{},{},{cumulative},{games_per_stage},{wins},{percent:.6}",
            pair.0, pair.1, layer_states[stage]
        )
        .unwrap();
        eprintln!(
            "stage={stage} layer=[{},{}] states={} win={percent:.3}%",
            pair.0, pair.1, layer_states[stage]
        );
    }
    output_file.flush().unwrap();
}

/// Stage 1: exact move regret for each heuristic, with no simulation.
///
/// For a sampled state, regret is the win-probability the mover gives up by
/// taking the heuristic's move instead of the optimal one. States are drawn
/// either uniformly over the table or from optimal-vs-optimal play; the two
/// distributions answer different questions and often disagree.
fn regret_report(
    model: &Path,
    output_dir: &Path,
    samples: usize,
    on_policy: bool,
    seed: u64,
    dump_features: bool,
    fitted_weights: Option<&Path>,
) {
    let lut = Lut::read(model);
    fs::create_dir_all(output_dir).unwrap();
    let distribution = if on_policy { "onpolicy" } else { "uniform" };

    // Collect the states to score.
    let mut rng = SplitMix64::new(seed);
    let mut states = Vec::with_capacity(samples);
    if on_policy {
        let mut moves = [0i8; 8];
        while states.len() < samples {
            let mut game = Game::initial(lut.rules);
            let mut plies = 0usize;
            while !game.finished && states.len() < samples {
                plies += 1;
                assert!(plies < 100_000);
                let roll = lut.rules.roll(&mut rng);
                let count = game.apply_roll(roll, &mut moves);
                if count == 0 {
                    continue;
                }
                if count > 1 {
                    states.push(game.clone());
                }
                let source = choose_optimal_move(&lut, &game, &moves[..count]);
                game.apply_move(source, lut.rules);
            }
        }
    } else {
        let mut moves = [0i8; 8];
        while states.len() < samples {
            let (key, _) = lut.key_value_at_global(rng.index(lut.total));
            let mut game = lut.encoding.decode(key);
            if game.finished {
                continue;
            }
            // Give the position a roll so there is a decision to make.
            let roll = lut.rules.roll(&mut rng);
            let count = game.apply_roll(roll, &mut moves);
            if count > 1 {
                states.push(game);
            }
        }
    }
    eprintln!("collected {} decision states ({distribution})", states.len());

    let mut summary = BufWriter::new(
        File::create(output_dir.join(format!("{}_regret_{distribution}.csv", lut.rules.name())))
            .unwrap(),
    );
    writeln!(summary, "heuristic,states,mean_regret,p95_regret,max_regret,agreement_pct").unwrap();

    // (name, weights); None means uniformly random move choice.
    let path_len = lut.rules.light_path().len() as f64;
    let mut ladder: Vec<(String, Option<[f64; WEIGHT_COUNT]>)> = Heuristic::all()
        .iter()
        .map(|h| {
            let weights = if *h == Heuristic::Random { None } else { Some(h.weights(path_len)) };
            (h.name().to_string(), weights)
        })
        .collect();
    // Weights fitted offline by scripts/fit_heuristic.py, one float per line in
    // FEATURE_NAMES order. This closes the loop: dump features, fit, feed back.
    if let Some(path) = fitted_weights {
        let text = fs::read_to_string(path).expect("failed to read weights file");
        let values: Vec<f64> = text
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(|line| line.parse::<f64>().expect("weights must be one float per line"))
            .collect();
        assert_eq!(
            values.len(),
            WEIGHT_COUNT,
            "expected {WEIGHT_COUNT} weights (features then intercept) in {}, found {}",
            path.display(),
            values.len()
        );
        let mut weights = [0.0; WEIGHT_COUNT];
        weights.copy_from_slice(&values);
        ladder.push(("fitted".to_string(), Some(weights)));
    }

    let mut moves = [0i8; 8];
    for (name, weights) in &ladder {
        let mut regrets = Vec::with_capacity(states.len());
        let mut agree = 0usize;
        let mut choice_rng = SplitMix64::new(seed ^ 0x5bd1_e995);
        for game in &states {
            let count = game.available_moves(&mut moves);
            debug_assert!(count > 1);
            let best = choose_optimal_move(&lut, game, &moves[..count]);
            let picked = choose_move_with(weights.as_ref(), &lut, game, &moves[..count], &mut choice_rng);
            let value_of = |source: i8| {
                let mut next = game.clone();
                next.apply_move(source, lut.rules);
                let light = lut.light_win_percent(&next);
                if game.is_light_turn { light } else { 100.0 - light }
            };
            let regret = (value_of(best) - value_of(picked)).max(0.0);
            if regret == 0.0 {
                agree += 1;
            }
            regrets.push(regret);
        }
        regrets.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mean = regrets.iter().sum::<f64>() / regrets.len() as f64;
        let p95 = regrets[(regrets.len() as f64 * 0.95) as usize % regrets.len()];
        let max = *regrets.last().unwrap();
        let agreement = 100.0 * agree as f64 / regrets.len() as f64;
        writeln!(
            summary,
            "{},{},{mean:.6},{p95:.6},{max:.6},{agreement:.4}",
            name,
            regrets.len()
        )
        .unwrap();
        eprintln!(
            "{:>12}: mean_regret={mean:.4} p95={p95:.4} max={max:.4} agreement={agreement:.2}%",
            name
        );
    }
    summary.flush().unwrap();

    // Feature matrix plus exact values, for fitting a linear model offline.
    if dump_features {
        let path = output_dir.join(format!("{}_features_{distribution}.csv", lut.rules.name()));
        let mut file = BufWriter::new(File::create(&path).unwrap());
        writeln!(file, "{},value_mover", FEATURE_NAMES.join(",")).unwrap();
        for game in &states {
            let f = features(&lut, game);
            let light = lut.light_win_percent(game);
            let mover = if game.is_light_turn { light } else { 100.0 - light };
            let row: Vec<String> = f.iter().map(|v| format!("{v:.6}")).collect();
            writeln!(file, "{},{mover:.9}", row.join(",")).unwrap();
        }
        file.flush().unwrap();
        eprintln!("wrote feature matrix to {}", path.display());
    }
}

fn usage() -> ! {
    eprintln!("usage:\n  royalur_analysis verify <model.rgu> [samples]\n  royalur_analysis analyze <model.rgu> <output-dir> [gap-states] [compare-states] [games-per-state] [games-per-epsilon]\n  royalur_analysis preflight-train <percent16-model.rgu> [samples]\n  royalur_analysis train-f64 <percent16-model.rgu> <output-f64.rgu> [tolerance] [max-iterations] [ondemand-jacobi|precomputed-gauss-seidel]\n  royalur_analysis bench-layer <percent16-model.rgu> <min-score> <max-score> [sweeps] [work-dir]\n  royalur_analysis simulate <model.rgu> <output-dir> <label> [compare-states] [games-per-state] [games-per-epsilon] [shard-seed]\n  royalur_analysis compare-checkpoint <checkpoint> <layer-dir> <f64-model.rgu>");
    std::process::exit(2);
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() < 3 {
        usage();
    }
    let command = args[1].as_str();
    let model = PathBuf::from(&args[2]);
    if command == "preflight-train" {
        let samples = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10_000);
        preflight_training(&model, samples);
        return;
    }
    if command == "dump-move-features" {
        if args.len() < 4 {
            usage();
        }
        let output = PathBuf::from(&args[3]);
        let samples = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(50_000);
        let seed = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0x6b2f_9e11_c40d_7a35);
        dump_move_features(&model, &output, samples, seed);
        return;
    }
    if command == "depth-regret" {
        if args.len() < 4 {
            usage();
        }
        let output = PathBuf::from(&args[3]);
        let samples = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(5_000);
        let max_depth = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(3);
        let weights = args.get(6).map(PathBuf::from);
        let seed = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(0x51ed_270b_44c9_0af3);
        depth_regret(&model, &output, samples, max_depth, weights.as_deref(), seed);
        return;
    }
    if command == "ordering-gram" {
        if args.len() < 4 {
            usage();
        }
        // stride 1 walks every state; a larger stride subsamples for a quick look.
        let stride = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1usize).max(1);
        ordering_gram(&model, Path::new(&args[3]), stride);
        return;
    }
    if command == "feature-gram" {
        if args.len() < 4 {
            usage();
        }
        feature_gram(&model, Path::new(&args[3]));
        return;
    }
    if command == "dump-moves" {
        if args.len() < 4 {
            usage();
        }
        let output = PathBuf::from(&args[3]);
        let samples = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(50_000);
        let on_policy = !matches!(args.get(5).map(String::as_str), Some("uniform"));
        let seed = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0x3ad9_51c7_60fe_2b44);
        dump_moves(&model, &output, samples, on_policy, seed);
        return;
    }
    if command == "stage-curve" {
        if args.len() < 4 {
            usage();
        }
        let output = PathBuf::from(&args[3]);
        let games = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(20_000);
        let seed = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0x2f6e_1c33_a7d5_04b9);
        stage_curve(&model, &output, games, seed);
        return;
    }
    if command == "regret" {
        if args.len() < 4 {
            usage();
        }
        let output_dir = PathBuf::from(&args[3]);
        let samples = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(200_000);
        let on_policy = matches!(args.get(5).map(String::as_str), Some("onpolicy"));
        let seed = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(0x71c3_9a5e_2b84_16df);
        let weights_path = args.get(7).map(PathBuf::from);
        regret_report(&model, &output_dir, samples, on_policy, seed, true, weights_path.as_deref());
        return;
    }
    if command == "simulate" {
        if args.len() < 5 {
            usage();
        }
        let lut = Lut::read(&model);
        let output_dir = PathBuf::from(&args[3]);
        let label = args[4].clone();
        let compare_states = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(100);
        let games_per_state = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(20_000);
        let games_per_epsilon = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(20_000);
        // The label doubles as the shard seed when it parses as a number, so a
        // Slurm array task id is enough to make a shard distinct.
        let shard_seed = args
            .get(8)
            .and_then(|s| s.parse::<u64>().ok())
            .or_else(|| label.parse::<u64>().ok())
            .unwrap_or(0)
            .wrapping_mul(0xa076_1d64_78bd_642f)
            .wrapping_add(0x9e37_79b9_7f4a_7c15);
        write_simulation_shard(
            &lut,
            &output_dir,
            &label,
            compare_states,
            games_per_state,
            games_per_epsilon,
            shard_seed,
        );
        return;
    }
    if command == "compare-checkpoint" {
        if args.len() < 5 {
            usage();
        }
        // model is args[2] = the checkpoint here.
        compare_checkpoint(&model, Path::new(&args[3]), Path::new(&args[4]));
        return;
    }
    if command == "bench-layer" {
        if args.len() < 5 {
            usage();
        }
        let min_score = args[3].parse::<u8>().expect("min-score must be an integer");
        let max_score = args[4].parse::<u8>().expect("max-score must be an integer");
        assert!(min_score <= max_score, "min-score must not exceed max-score");
        let sweeps = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(10);
        let work_dir = args
            .get(6)
            .map(PathBuf::from)
            .unwrap_or_else(|| model.parent().unwrap_or(Path::new(".")).join("bench"));
        bench_layer(&model, (min_score, max_score), sweeps, &work_dir);
        return;
    }
    if command == "train-f64" {
        if args.len() < 4 {
            usage();
        }
        let output = PathBuf::from(&args[3]);
        let tolerance = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1e-12);
        let max_iterations = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(10_000);
        let strategy = match args.get(6).map(String::as_str) {
            None | Some("ondemand-jacobi") => Strategy::OnDemandJacobi,
            Some("precomputed-gauss-seidel") => Strategy::PrecomputedGaussSeidel,
            Some(other) => {
                eprintln!("unknown strategy: {other}");
                usage();
            }
        };
        let init = match args.get(7).map(String::as_str) {
            None | Some("published") => Init::Published,
            Some("naive") => Init::Naive,
            Some(other) => {
                eprintln!("unknown init: {other}");
                usage();
            }
        };
        train_f64(&model, &output, tolerance, max_iterations, strategy, init);
        return;
    }
    let lut = Lut::read(&model);
    match command {
        "verify" => {
            let samples = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(10_000);
            verify(&lut, samples);
        }
        "analyze" => {
            if args.len() < 4 {
                usage();
            }
            let output_dir = PathBuf::from(&args[3]);
            fs::create_dir_all(&output_dir).unwrap();
            let gap_states = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(1_000_000);
            let compare_states = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(100);
            let games_per_state = args.get(6).and_then(|s| s.parse().ok()).unwrap_or(20_000);
            let games_per_epsilon = args.get(7).and_then(|s| s.parse().ok()).unwrap_or(20_000);
            let name = lut.rules.name();
            write_gap_sample(&lut, &output_dir.join(format!("{name}_gaps.csv")), gap_states, 0x91a2_b3c4_d5e6_f701);
            write_compare(&lut, &output_dir.join(format!("{name}_compare.csv")), compare_states, games_per_state, 0x1234_5678_9abc_def0);
            write_epsilon(&lut, &output_dir.join(format!("{name}_epsilon.csv")), games_per_epsilon, 0xfedc_ba98_7654_3210);
        }
        _ => usage(),
    }
}
