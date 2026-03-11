from __future__ import annotations

from mcir.builder import MCIRBuilder
from mcir.module import MCModule


def run_streaming_lowering_pass(module: MCModule) -> MCModule:
    b = MCIRBuilder()

    for region in module.regions:
        if region.name != "attention_region":
            continue

        q = region.inputs[0]
        k = region.inputs[1]
        v = region.inputs[2]
        o = region.outputs[0]

        stream = b.streaming_region("attention_stream", stream_axis="sequence")
        tile = b.tile_region("attention_tile", tile_m=128, tile_n=64, tile_k=64)

        q_tile = b.value("Q_tile", (128, 64), q.dtype, residency="shared")
        k_tile = b.value("K_tile", (64, 64), k.dtype, residency="shared")
        v_tile = b.value("V_tile", (64, 64), v.dtype, residency="shared")

        score_frag = b.value("score_frag", (128, 64), "fp32", residency="register")
        softmax_max = b.value("softmax_max", (128,), "fp32", residency="register")
        softmax_sum = b.value("softmax_sum", (128,), "fp32", residency="register")
        output_acc = b.value("output_acc", (128, 64), "fp32", residency="register")
        o_tile = b.value("O_tile", (128, 64), o.dtype, residency="global")

        tile.nodes.append(
            b.node("load_q", "load_tile", inputs=[q], outputs=[q_tile], source="Q")
        )
        tile.nodes.append(
            b.node("load_k", "load_tile", inputs=[k], outputs=[k_tile], source="K")
        )
        tile.nodes.append(
            b.node("load_v", "load_tile", inputs=[v], outputs=[v_tile], source="V")
        )
        tile.nodes.append(
            b.node(
                "compute_score",
                "compute_score",
                inputs=[q_tile, k_tile],
                outputs=[score_frag],
            )
        )
        tile.nodes.append(
            b.node(
                "update_softmax",
                "update_softmax",
                inputs=[score_frag],
                outputs=[softmax_max, softmax_sum],
                online=True,
            )
        )
        tile.nodes.append(
            b.node(
                "accumulate_output",
                "accumulate_output",
                inputs=[score_frag, v_tile],
                outputs=[output_acc],
            )
        )
        tile.nodes.append(
            b.node("store_o", "store_tile", inputs=[output_acc], outputs=[o_tile], target="O")
        )

        stream.inputs.extend([q, k, v])
        stream.outputs.append(o)
        stream.subregions.append(tile)

        region.subregions.append(stream)
        region.attrs["lowered_to_streaming"] = True

    return module