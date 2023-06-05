#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Segment {
    left: i32,
    right: i32,
}

impl Segment {
    fn left(&self) -> i32 {
        self.left
    }
    fn right(&self) -> i32 {
        self.right
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Label {
    A,
    B,
    C,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct LabelledSegment {
    segment: Segment,
    label: Label,
}

impl LabelledSegment {
    fn left(&self) -> i32 {
        self.segment.left()
    }
    fn right(&self) -> i32 {
        self.segment.right()
    }
}

fn look_ahead(other_data: &[LabelledSegment], left: i32, right: i32) -> usize {
    let mut num_overlaps = 1;

    for j in other_data {
        if j.right() > left && right > j.left() {
            num_overlaps += 1;
        } else {
            break;
        }
    }

    num_overlaps
}

// Precondition: other_data must be SORTED
// NOTE: this is basically what we do over in tskit.
// The difference is that, mentally, we are comparing
// current ancestry to existing ancestry and recording the overlap.
// However, this is not "quite" what we want to do.
// Rather, we need to find stuff that doesn't overlap
// and keep it if it is useful to do so.
fn calculate_overlaps(
    data: &[LabelledSegment],
    other_data: &[LabelledSegment],
) -> Vec<LabelledSegment> {
    let mut overlaps = vec![];

    let mut queue = vec![];
    let mut d = 0;
    while d < data.len() {
        queue.push(data[d]);
        for o in other_data {
            if o.right() > data[d].left() && data[d].right() > o.left() {
                let left = std::cmp::max(data[d].left(), o.left());
                let right = std::cmp::min(data[d].right(), o.right());
                queue.push(LabelledSegment {
                    segment: Segment { left, right },
                    label: o.label,
                });
            }
        }
        // NOTE: it is possible that data contains multiple
        // identical entries, meaning overlapping the exact same
        // intervals but for different child nodes.
        // Production code will need extra testing to make sure
        // that the update is sane.
        let update = data.iter().skip(d).take_while(|&&x| x == data[d]).count();
        d += update + 1;
    }
    assert!(
        queue.windows(2).all(|w| w[0].left() <= w[1].left()),
        "queue not sorted: {queue:?}"
    );

    let mut local_overlaps: Vec<LabelledSegment> = vec![];
    let n = queue.len();
    queue.push(LabelledSegment {
        segment: Segment {
            left: i32::MAX,
            right: i32::MAX,
        },
        label: Label::C,
    });
    let mut right_position = queue[0].right();
    let mut i = 0;
    while i < n {
        let mut left_position = right_position;
        local_overlaps.retain(|x| x.right() > left_position);
        if local_overlaps.is_empty() {
            left_position = queue[i].left();
        }
        while i < n && queue[i].left() == left_position {
            let x = queue[i];
            if x.label != data[0].label {
                right_position = std::cmp::min(right_position, x.right());
                local_overlaps.push(x);
            }
            i += 1;
        }
        i -= 1;
        if !local_overlaps.is_empty() {
            right_position = local_overlaps.iter().map(|o| o.right()).min().unwrap();
        }
        right_position = std::cmp::min(right_position, queue[i + 1].left());
        i += 1;
        //overlaps.extend_from_slice(&local_overlaps);
        for x in &local_overlaps {
            overlaps.push(LabelledSegment {
                segment: Segment {
                    left: left_position,
                    right: right_position,
                },
                label: x.label,
            })
        }
    }
    if !local_overlaps.is_empty() {
        let left_position = right_position;
        local_overlaps.retain(|x| x.right() > left_position);
        if !(local_overlaps.is_empty()) {
            right_position = local_overlaps.iter().map(|o| o.right()).min().unwrap();
            for x in &local_overlaps {
                overlaps.push(LabelledSegment {
                    segment: Segment {
                        left: left_position,
                        right: right_position,
                    },
                    label: x.label,
                })
            }
        }
    }
    // NOTE: this only necessary
    // because of how we are writing tests
    overlaps.sort_unstable();
    overlaps
}

#[test]
fn test_overlap_0() {
    let data = [LabelledSegment {
        segment: Segment { left: 0, right: 10 },
        label: Label::C,
    }];

    let other_data = [
        LabelledSegment {
            segment: Segment { left: 0, right: 10 },
            label: Label::A,
        },
        LabelledSegment {
            segment: Segment { left: 0, right: 10 },
            label: Label::A,
        },
    ];

    calculate_overlaps(&data, &other_data);
}

#[test]
fn test_overlap_1() {
    let data = [LabelledSegment {
        segment: Segment { left: 5, right: 9 },
        label: Label::C,
    }];

    let other_data = [
        LabelledSegment {
            segment: Segment { left: 3, right: 7 },
            label: Label::A,
        },
        LabelledSegment {
            segment: Segment { left: 4, right: 8 },
            label: Label::B,
        },
    ];

    let overlaps = calculate_overlaps(&data, &other_data);
    assert_eq!(
        overlaps,
        [
            LabelledSegment {
                segment: Segment { left: 5, right: 7 },
                label: Label::A,
            },
            LabelledSegment {
                segment: Segment { left: 5, right: 7 },
                label: Label::B,
            },
            LabelledSegment {
                segment: Segment { left: 7, right: 8 },
                label: Label::B,
            },
        ]
    )
}

#[test]
fn test_overlap_2() {
    let data = [LabelledSegment {
        segment: Segment { left: 5, right: 6 },
        label: Label::C,
    }];

    let other_data = [
        LabelledSegment {
            segment: Segment { left: 3, right: 7 },
            label: Label::A,
        },
        LabelledSegment {
            segment: Segment { left: 4, right: 8 },
            label: Label::B,
        },
    ];

    let overlaps = calculate_overlaps(&data, &other_data);
    assert_eq!(
        overlaps,
        [
            LabelledSegment {
                segment: Segment { left: 5, right: 6 },
                label: Label::A,
            },
            LabelledSegment {
                segment: Segment { left: 5, right: 6 },
                label: Label::B,
            },
        ]
    )
}

#[test]
fn test_overlap_3() {
    let data = [LabelledSegment {
        segment: Segment { left: 0, right: 10 },
        label: Label::C,
    }];

    let other_data = [
        LabelledSegment {
            segment: Segment { left: 3, right: 4 },
            label: Label::A,
        },
        LabelledSegment {
            segment: Segment { left: 3, right: 5 },
            label: Label::B,
        },
        LabelledSegment {
            segment: Segment { left: 4, right: 6 },
            label: Label::A,
        },
    ];

    let overlaps = calculate_overlaps(&data, &other_data);
    assert_eq!(
        overlaps,
        [
            LabelledSegment {
                segment: Segment { left: 3, right: 4 },
                label: Label::A,
            },
            LabelledSegment {
                segment: Segment { left: 3, right: 4 },
                label: Label::B,
            },
            LabelledSegment {
                segment: Segment { left: 4, right: 5 },
                label: Label::A,
            },
            LabelledSegment {
                segment: Segment { left: 4, right: 5 },
                label: Label::B,
            },
            LabelledSegment {
                segment: Segment { left: 5, right: 6 },
                label: Label::A,
            },
        ]
    )
}

#[test]
fn test_overlap_4() {
    let data = [LabelledSegment {
        segment: Segment { left: 0, right: 10 },
        label: Label::C,
    }];

    let other_data = [
        LabelledSegment {
            segment: Segment { left: 3, right: 4 },
            label: Label::A,
        },
        LabelledSegment {
            segment: Segment { left: 3, right: 5 },
            label: Label::B,
        },
        LabelledSegment {
            segment: Segment { left: 7, right: 9 },
            label: Label::A,
        },
        LabelledSegment {
            segment: Segment { left: 8, right: 10 },
            label: Label::B,
        },
    ];

    let overlaps = calculate_overlaps(&data, &other_data);
    assert_eq!(
        overlaps,
        [
            LabelledSegment {
                segment: Segment { left: 3, right: 4 },
                label: Label::A,
            },
            LabelledSegment {
                segment: Segment { left: 3, right: 4 },
                label: Label::B,
            },
            LabelledSegment {
                segment: Segment { left: 4, right: 5 },
                label: Label::B,
            },
            LabelledSegment {
                segment: Segment { left: 7, right: 8 },
                label: Label::A,
            },
            LabelledSegment {
                segment: Segment { left: 8, right: 9 },
                label: Label::A,
            },
            LabelledSegment {
                segment: Segment { left: 8, right: 9 },
                label: Label::B,
            },
            LabelledSegment {
                segment: Segment { left: 9, right: 10 },
                label: Label::B,
            },
        ]
    )
}

#[test]
fn test_overlap_5() {
    let data = [LabelledSegment {
        segment: Segment { left: 0, right: 10 },
        label: Label::C,
    }];

    let other_data = [
        LabelledSegment {
            segment: Segment { left: 3, right: 5 },
            label: Label::B,
        },
        LabelledSegment {
            segment: Segment { left: 7, right: 9 },
            label: Label::A,
        },
    ];
    let overlaps = calculate_overlaps(&data, &other_data);
    assert_eq!(
        overlaps,
        [
            LabelledSegment {
                segment: Segment { left: 3, right: 5 },
                label: Label::B,
            },
            LabelledSegment {
                segment: Segment { left: 7, right: 9 },
                label: Label::A,
            },
        ]
    );
}

#[test]
fn test_overlap_5b() {
    let data = [
        LabelledSegment {
            segment: Segment { left: 0, right: 10 },
            label: Label::C,
        },
        LabelledSegment {
            segment: Segment { left: 0, right: 10 },
            label: Label::C,
        },
    ];

    let other_data = [
        LabelledSegment {
            segment: Segment { left: 3, right: 5 },
            label: Label::B,
        },
        LabelledSegment {
            segment: Segment { left: 7, right: 9 },
            label: Label::A,
        },
    ];
    let overlaps = calculate_overlaps(&data, &other_data);
    assert_eq!(
        overlaps,
        [
            LabelledSegment {
                segment: Segment { left: 3, right: 5 },
                label: Label::B,
            },
            LabelledSegment {
                segment: Segment { left: 7, right: 9 },
                label: Label::A,
            },
        ]
    );
}
#[test]
fn test_overlap_6() {
    let data = [LabelledSegment {
        segment: Segment { left: 0, right: 10 },
        label: Label::C,
    }];

    let other_data = [
        LabelledSegment {
            segment: Segment { left: 3, right: 5 },
            label: Label::B,
        },
        LabelledSegment {
            segment: Segment { left: 4, right: 9 },
            label: Label::A,
        },
    ];
    let overlaps = calculate_overlaps(&data, &other_data);
    assert_eq!(
        overlaps,
        [
            LabelledSegment {
                segment: Segment { left: 3, right: 4 },
                label: Label::B,
            },
            LabelledSegment {
                segment: Segment { left: 4, right: 5 },
                label: Label::A,
            },
            LabelledSegment {
                segment: Segment { left: 4, right: 5 },
                label: Label::B,
            },
            LabelledSegment {
                segment: Segment { left: 5, right: 9 },
                label: Label::A,
            },
        ]
    );
}

#[test]
fn test_sorting() {
    {
        let mut data = [
            LabelledSegment {
                segment: Segment { left: 5, right: 6 },
                label: Label::A,
            },
            LabelledSegment {
                segment: Segment { left: 4, right: 6 },
                label: Label::A,
            },
        ];
        data.sort_unstable();
        assert_eq!(
            data,
            [
                LabelledSegment {
                    segment: Segment { left: 4, right: 6 },
                    label: Label::A
                },
                LabelledSegment {
                    segment: Segment { left: 5, right: 6 },
                    label: Label::A
                },
            ]
        )
    }

    {
        let mut data = [
            LabelledSegment {
                segment: Segment { left: 5, right: 6 },
                label: Label::B,
            },
            LabelledSegment {
                segment: Segment { left: 5, right: 6 },
                label: Label::A,
            },
        ];
        data.sort_unstable();
        assert_eq!(
            data,
            [
                LabelledSegment {
                    segment: Segment { left: 5, right: 6 },
                    label: Label::A
                },
                LabelledSegment {
                    segment: Segment { left: 5, right: 6 },
                    label: Label::B
                },
            ]
        )
    }

    {
        let mut data = [
            LabelledSegment {
                segment: Segment { left: 5, right: 8 },
                label: Label::A,
            },
            LabelledSegment {
                segment: Segment { left: 5, right: 6 },
                label: Label::A,
            },
        ];
        data.sort_unstable();
        assert_eq!(
            data,
            [
                LabelledSegment {
                    segment: Segment { left: 5, right: 6 },
                    label: Label::A
                },
                LabelledSegment {
                    segment: Segment { left: 5, right: 8 },
                    label: Label::A
                },
            ]
        )
    }
}
