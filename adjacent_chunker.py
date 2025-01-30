import spacy
from encoder import Encoder
import heapq
import torch


def _get_similarities(encoder, batch):
    embeddings = encoder.encode_n(batch)
    similarities = encoder.get_adjacent_similarity_statistics(embeddings)
    return similarities


class AdjacentChunker:
    def __init__(self, no_first_passes, no_second_passes, k1, k2, batch_size):
        self.k1 = k1
        self.k2 = k2
        self.no_first_passes = no_first_passes
        self.no_second_passes = no_second_passes
        self.device = 'cuda'
        self.batch_size = batch_size
        self.sentence_splitter = spacy.load('en_core_web_sm')
        self.small_encoder = Encoder('sentence-transformers/all-MiniLM-L6-v2')
        self.large_encoder = Encoder('mixedbread-ai/mxbai-embed-large-v1')

    def chunk_adjacent(self, text):
        chunks = self.sentence_splitter(text)
        chunks = [sent.text for sent in chunks.sents]
        for i in range(self.no_first_passes):
            similarities, threshold = (self._get_statistical_chunking_metrics
                                       (chunks, self.small_encoder, self.k1, self.device, self.batch_size))
            chunks = self._merge_adjacent_chunks(chunks, threshold, similarities)

        for i in range(self.no_second_passes):
            similarities, threshold = (self._get_statistical_chunking_metrics
                                       (chunks, self.large_encoder, self.k2, self.device, self.batch_size))
            chunks = self._merge_adjacent_chunks(chunks, threshold, similarities)

        return chunks

    @staticmethod
    def _merge_adjacent_chunks(chunks, threshold, similarities):
        priority_queue = []
        for i in range(len(similarities)):
            similarity = similarities[i]
            heapq.heappush(priority_queue, (-similarity, (i, i+1)))

        was_merged = set()
        idx_to_merged = {}
        while priority_queue:
            similarity, (i, j) = heapq.heappop(priority_queue)

            if -similarity < threshold:
                break

            if i in was_merged or j in was_merged:
                continue

            was_merged.add(i)
            was_merged.add(j)
            idx_to_merged[i] = chunks[i] + ' ' + chunks[j]

        new_chunks = []
        for i in range(len(chunks)):
            if i in idx_to_merged:
                new_chunks.append(idx_to_merged[i])
            elif i in was_merged:
                continue
            else:
                new_chunks.append(chunks[i])

        return new_chunks

    @staticmethod
    def _get_statistical_chunking_metrics(chunks, encoder, k, device, batch_size):
        def batch_adjacent(chunk_batches, len_batch):
            for i in range(0, len(chunk_batches) - 1, len_batch):
                yield chunk_batches[i: min(i + len_batch + 1, len(chunk_batches))]

        similarity_tensor = torch.empty(0).to(device)
        for batch in batch_adjacent(chunks, batch_size):
            batch_similarity = _get_similarities(encoder, batch)
            similarity_tensor = torch.cat((similarity_tensor, batch_similarity))

        def _get_mad_med(tensor):
            me = torch.median(tensor).item()
            ma = torch.median(torch.abs(tensor - me)).item()
            return ma, me

        mad, median = _get_mad_med(similarity_tensor)

        return similarity_tensor.tolist(), median + mad * k
