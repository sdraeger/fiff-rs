use super::constants::*;
use super::tag::DirEntry;
/// FIFF Tree/Directory parsing
/// Based on MNE-Python's _fiff/tree.py
use std::io::{Read, Seek};

/// FIFF tree node
#[derive(Debug, Clone)]
pub struct TreeNode {
    pub block: i32,
    pub directory: Vec<DirEntry>,
    pub nent: usize,
    pub children: Vec<TreeNode>,
}

impl TreeNode {
    pub fn new(block: i32) -> Self {
        TreeNode {
            block,
            directory: Vec::new(),
            nent: 0,
            children: Vec::new(),
        }
    }
}

/// Find nodes with a specific block type in the tree
pub fn dir_tree_find(node: &TreeNode, block_type: i32) -> Vec<&TreeNode> {
    let mut results = Vec::new();

    if node.block == block_type {
        results.push(node);
    }

    for child in &node.children {
        results.extend(dir_tree_find(child, block_type));
    }

    results
}

/// Build a tree from directory entries
pub fn build_tree<R: Read + Seek>(
    reader: &mut R,
    directory: Vec<DirEntry>,
) -> std::io::Result<TreeNode> {
    let mut root = TreeNode::new(0);
    // Don't initialize root.directory - only add top-level entries in the loop

    let mut current_child: Option<TreeNode> = None;
    let mut block_stack: Vec<TreeNode> = Vec::new();

    for entry in &directory {
        // Block start (kind = 104)
        if entry.kind == FIFF_BLOCK_START {
            // Read the tag data to get block type
            let tag = super::tag::Tag::read_at(reader, entry.pos)?;
            let block_type = tag.as_i32().unwrap_or(0);

            // Save current child if exists
            if let Some(child) = current_child.take() {
                block_stack.push(child);
            }

            current_child = Some(TreeNode::new(block_type));
        }
        // Block end (kind = 105)
        else if entry.kind == FIFF_BLOCK_END {
            if let Some(child) = current_child.take() {
                // If there's a parent block on the stack, add to it
                if let Some(mut parent) = block_stack.pop() {
                    parent.children.push(child);
                    current_child = Some(parent);
                } else {
                    // Otherwise add to root
                    root.children.push(child);
                }
            }
        }
        // Regular tag entry - add to current block
        else if let Some(ref mut child) = current_child {
            child.directory.push(entry.clone());
            child.nent += 1;
        } else {
            // Top-level tag (not in any block)
            root.directory.push(entry.clone());
            root.nent += 1;
        }
    }

    // Add any remaining child to root
    if let Some(child) = current_child {
        root.children.push(child);
    }

    // Add any remaining blocks from stack
    while let Some(child) = block_stack.pop() {
        root.children.push(child);
    }

    eprintln!(
        "Built tree with {} top-level entries and {} child blocks",
        root.nent,
        root.children.len()
    );

    Ok(root)
}

#[cfg(test)]
mod tests {
    use super::*;
    use byteorder::{BigEndian, WriteBytesExt};
    use std::io::Cursor;

    fn create_test_dir_entry(kind: i32, type_: i32, size: i32, pos: u64) -> DirEntry {
        DirEntry {
            kind,
            type_,
            size,
            pos,
        }
    }

    fn create_test_tag_bytes(kind: i32, type_: i32, size: i32, data: &[u8]) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.write_i32::<BigEndian>(kind).unwrap();
        bytes.write_i32::<BigEndian>(type_).unwrap();
        bytes.write_i32::<BigEndian>(size).unwrap();
        bytes.write_i32::<BigEndian>(0).unwrap(); // next pointer
        bytes.extend_from_slice(data);
        bytes
    }

    #[test]
    fn test_tree_node_new() {
        let node = TreeNode::new(FIFFB_MEAS);
        assert_eq!(node.block, FIFFB_MEAS);
        assert_eq!(node.directory.len(), 0);
        assert_eq!(node.nent, 0);
        assert_eq!(node.children.len(), 0);
    }

    #[test]
    fn test_dir_tree_find_empty() {
        let root = TreeNode::new(0);
        let results = dir_tree_find(&root, FIFFB_MEAS);
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_dir_tree_find_root_match() {
        let root = TreeNode::new(FIFFB_MEAS);
        let results = dir_tree_find(&root, FIFFB_MEAS);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].block, FIFFB_MEAS);
    }

    #[test]
    fn test_dir_tree_find_child_match() {
        let mut root = TreeNode::new(0);
        let child1 = TreeNode::new(FIFFB_MEAS);
        let child2 = TreeNode::new(FIFFB_RAW_DATA);
        root.children.push(child1);
        root.children.push(child2);

        let results = dir_tree_find(&root, FIFFB_MEAS);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].block, FIFFB_MEAS);
    }

    #[test]
    fn test_dir_tree_find_nested() {
        let mut root = TreeNode::new(0);
        let mut child = TreeNode::new(FIFFB_MEAS);
        let grandchild1 = TreeNode::new(FIFFB_RAW_DATA);
        let grandchild2 = TreeNode::new(FIFFB_RAW_DATA);
        child.children.push(grandchild1);
        child.children.push(grandchild2);
        root.children.push(child);

        let results = dir_tree_find(&root, FIFFB_RAW_DATA);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].block, FIFFB_RAW_DATA);
        assert_eq!(results[1].block, FIFFB_RAW_DATA);
    }

    #[test]
    fn test_dir_tree_find_multiple_levels() {
        let mut root = TreeNode::new(0); // Root node
        let mut level1 = TreeNode::new(FIFFB_MEAS);
        let level2a = TreeNode::new(FIFFB_RAW_DATA);
        let level2b = TreeNode::new(FIFFB_RAW_DATA);
        level1.children.push(level2a);
        level1.children.push(level2b);
        root.children.push(level1);

        // Find FIFFB_MEAS
        let meas_results = dir_tree_find(&root, FIFFB_MEAS);
        assert_eq!(meas_results.len(), 1);

        // Find FIFFB_RAW_DATA
        let raw_results = dir_tree_find(&root, FIFFB_RAW_DATA);
        assert_eq!(raw_results.len(), 2);

        // Find root with block type 0
        let root_results = dir_tree_find(&root, 0);
        assert_eq!(root_results.len(), 1);
    }

    #[test]
    fn test_build_tree_empty() {
        let directory = Vec::new();
        let mut cursor = Cursor::new(Vec::new());
        let root = build_tree(&mut cursor, directory).unwrap();

        assert_eq!(root.block, 0);
        assert_eq!(root.nent, 0);
        assert_eq!(root.children.len(), 0);
    }

    #[test]
    fn test_build_tree_top_level_tags() {
        // Create directory with top-level tags (no blocks)
        let directory = vec![
            create_test_dir_entry(FIFF_NCHAN, FIFFT_INT, 4, 0),
            create_test_dir_entry(FIFF_SFREQ, FIFFT_FLOAT, 4, 16),
        ];

        let mut cursor = Cursor::new(Vec::new());
        let root = build_tree(&mut cursor, directory).unwrap();

        assert_eq!(root.block, 0);
        assert_eq!(root.nent, 2);
        assert_eq!(root.directory.len(), 2);
        assert_eq!(root.children.len(), 0);
    }

    #[test]
    fn test_build_tree_single_block() {
        // Create a simple block structure:
        // BLOCK_START(FIFFB_MEAS) -> FIFF_NCHAN -> BLOCK_END
        let mut file_bytes = Vec::new();

        // BLOCK_START tag with FIFFB_MEAS block type
        let mut block_data = Vec::new();
        block_data.write_i32::<BigEndian>(FIFFB_MEAS).unwrap();
        file_bytes.extend_from_slice(&create_test_tag_bytes(
            FIFF_BLOCK_START,
            FIFFT_INT,
            4,
            &block_data,
        ));

        // Regular tag inside block
        let pos_nchan = file_bytes.len() as u64;
        let mut nchan_data = Vec::new();
        nchan_data.write_i32::<BigEndian>(100).unwrap();
        file_bytes.extend_from_slice(&create_test_tag_bytes(
            FIFF_NCHAN,
            FIFFT_INT,
            4,
            &nchan_data,
        ));

        // BLOCK_END tag
        let pos_end = file_bytes.len() as u64;
        file_bytes.extend_from_slice(&create_test_tag_bytes(FIFF_BLOCK_END, FIFFT_INT, 0, &[]));

        let directory = vec![
            create_test_dir_entry(FIFF_BLOCK_START, FIFFT_INT, 4, 0),
            create_test_dir_entry(FIFF_NCHAN, FIFFT_INT, 4, pos_nchan),
            create_test_dir_entry(FIFF_BLOCK_END, FIFFT_INT, 0, pos_end),
        ];

        let mut cursor = Cursor::new(file_bytes);
        let root = build_tree(&mut cursor, directory).unwrap();

        assert_eq!(root.children.len(), 1);
        assert_eq!(root.children[0].block, FIFFB_MEAS);
        assert_eq!(root.children[0].nent, 1);
        assert_eq!(root.children[0].directory.len(), 1);
        assert_eq!(root.children[0].directory[0].kind, FIFF_NCHAN);
    }

    #[test]
    fn test_build_tree_nested_blocks() {
        // Create nested block structure:
        // BLOCK_START(FIFFB_MEAS)
        //   BLOCK_START(FIFFB_RAW_DATA)
        //     FIFF_DATA_BUFFER
        //   BLOCK_END
        // BLOCK_END
        let mut file_bytes = Vec::new();

        // Outer BLOCK_START (FIFFB_MEAS)
        let mut meas_block = Vec::new();
        meas_block.write_i32::<BigEndian>(FIFFB_MEAS).unwrap();
        file_bytes.extend_from_slice(&create_test_tag_bytes(
            FIFF_BLOCK_START,
            FIFFT_INT,
            4,
            &meas_block,
        ));

        // Inner BLOCK_START (FIFFB_RAW_DATA)
        let pos_raw_start = file_bytes.len() as u64;
        let mut raw_block = Vec::new();
        raw_block.write_i32::<BigEndian>(FIFFB_RAW_DATA).unwrap();
        file_bytes.extend_from_slice(&create_test_tag_bytes(
            FIFF_BLOCK_START,
            FIFFT_INT,
            4,
            &raw_block,
        ));

        // Data tag inside inner block
        let pos_data = file_bytes.len() as u64;
        let data = vec![1u8, 2, 3, 4];
        file_bytes.extend_from_slice(&create_test_tag_bytes(
            FIFF_DATA_BUFFER,
            FIFFT_FLOAT,
            4,
            &data,
        ));

        // Inner BLOCK_END
        let pos_raw_end = file_bytes.len() as u64;
        file_bytes.extend_from_slice(&create_test_tag_bytes(FIFF_BLOCK_END, FIFFT_INT, 0, &[]));

        // Outer BLOCK_END
        let pos_meas_end = file_bytes.len() as u64;
        file_bytes.extend_from_slice(&create_test_tag_bytes(FIFF_BLOCK_END, FIFFT_INT, 0, &[]));

        let directory = vec![
            create_test_dir_entry(FIFF_BLOCK_START, FIFFT_INT, 4, 0),
            create_test_dir_entry(FIFF_BLOCK_START, FIFFT_INT, 4, pos_raw_start),
            create_test_dir_entry(FIFF_DATA_BUFFER, FIFFT_FLOAT, 4, pos_data),
            create_test_dir_entry(FIFF_BLOCK_END, FIFFT_INT, 0, pos_raw_end),
            create_test_dir_entry(FIFF_BLOCK_END, FIFFT_INT, 0, pos_meas_end),
        ];

        let mut cursor = Cursor::new(file_bytes);
        let root = build_tree(&mut cursor, directory).unwrap();

        // Root should have one child (FIFFB_MEAS)
        assert_eq!(root.children.len(), 1);
        assert_eq!(root.children[0].block, FIFFB_MEAS);

        // FIFFB_MEAS should have one child (FIFFB_RAW_DATA)
        assert_eq!(root.children[0].children.len(), 1);
        assert_eq!(root.children[0].children[0].block, FIFFB_RAW_DATA);

        // FIFFB_RAW_DATA should have one directory entry
        assert_eq!(root.children[0].children[0].nent, 1);
        assert_eq!(
            root.children[0].children[0].directory[0].kind,
            FIFF_DATA_BUFFER
        );
    }

    #[test]
    fn test_build_tree_multiple_siblings() {
        // Create structure with sibling blocks:
        // BLOCK_START(FIFFB_RAW_DATA) -> BLOCK_END
        // BLOCK_START(FIFFB_RAW_DATA) -> BLOCK_END
        let mut file_bytes = Vec::new();

        // First block
        let mut block1 = Vec::new();
        block1.write_i32::<BigEndian>(FIFFB_RAW_DATA).unwrap();
        file_bytes.extend_from_slice(&create_test_tag_bytes(
            FIFF_BLOCK_START,
            FIFFT_INT,
            4,
            &block1,
        ));
        let pos_end1 = file_bytes.len() as u64;
        file_bytes.extend_from_slice(&create_test_tag_bytes(FIFF_BLOCK_END, FIFFT_INT, 0, &[]));

        // Second block
        let pos_start2 = file_bytes.len() as u64;
        let mut block2 = Vec::new();
        block2.write_i32::<BigEndian>(FIFFB_RAW_DATA).unwrap();
        file_bytes.extend_from_slice(&create_test_tag_bytes(
            FIFF_BLOCK_START,
            FIFFT_INT,
            4,
            &block2,
        ));
        let pos_end2 = file_bytes.len() as u64;
        file_bytes.extend_from_slice(&create_test_tag_bytes(FIFF_BLOCK_END, FIFFT_INT, 0, &[]));

        let directory = vec![
            create_test_dir_entry(FIFF_BLOCK_START, FIFFT_INT, 4, 0),
            create_test_dir_entry(FIFF_BLOCK_END, FIFFT_INT, 0, pos_end1),
            create_test_dir_entry(FIFF_BLOCK_START, FIFFT_INT, 4, pos_start2),
            create_test_dir_entry(FIFF_BLOCK_END, FIFFT_INT, 0, pos_end2),
        ];

        let mut cursor = Cursor::new(file_bytes);
        let root = build_tree(&mut cursor, directory).unwrap();

        // Root should have two children
        assert_eq!(root.children.len(), 2);
        assert_eq!(root.children[0].block, FIFFB_RAW_DATA);
        assert_eq!(root.children[1].block, FIFFB_RAW_DATA);
    }
}
